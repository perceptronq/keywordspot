import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels, sample_rate=16000, duration=1):
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.desired_length = int(duration * sample_rate)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if waveform.shape[1] < self.desired_length:
            padding = self.desired_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.desired_length]

        return waveform, self.labels[idx]


class KeywordSpottingDNN(nn.Module):
    def __init__(self, num_classes=10, input_size=1024):
        super().__init__()

        self.dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.dnn(x)


class KeywordSpottingTrainer:
    def __init__(self, dataset_path, output_dir='./models'):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=480,
            hop_length=160,
            n_mels=64
        )

    def prepare_data(self):
        audio_paths = []
        labels = []

        for label in os.listdir(self.dataset_path):
            label_dir = os.path.join(self.dataset_path, label)
            if os.path.isdir(label_dir):
                for audio_file in os.listdir(label_dir):
                    if audio_file.endswith('.wav'):
                        audio_paths.append(os.path.join(label_dir, audio_file))
                        labels.append(label)

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        train_paths, test_paths, train_labels, test_labels = train_test_split(
            audio_paths, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )

        train_dataset = AudioDataset(train_paths, train_labels)
        test_dataset = AudioDataset(test_paths, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        return train_loader, test_loader, label_encoder

    def train(self, train_loader, test_loader, label_encoder, epochs=100):
        model = KeywordSpottingDNN(num_classes=len(
            label_encoder.classes_)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'max', patience=5)

        best_accuracy = 0.0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for waveforms, labels in train_loader:
                waveforms, labels = waveforms.to(
                    self.device), labels.to(self.device)
                specs = self.mel_transform(waveforms)

                # Flatten the mel spectrogram output to match the DNN input
                specs = specs.view(specs.size(0), -1)

                optimizer.zero_grad()
                outputs = model(specs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for waveforms, labels in test_loader:
                    waveforms, labels = waveforms.to(
                        self.device), labels.to(self.device)
                    specs = self.mel_transform(waveforms)

                    specs = specs.view(specs.size(0), -1)

                    outputs = model(specs)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            train_acc = 100. * correct / total
            val_acc = 100. * val_correct / val_total

            print(f'Epoch: {epoch+1}/{epochs}')
            print(f'Loss: {running_loss/len(train_loader):.4f}')
            print(f'Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')

            scheduler.step(val_acc)

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(model.state_dict(),
                           self.output_dir / 'dnn_model.pth')

        return model


def main():
    DATASET_PATH = 'C:/Users/sures/Downloads/keyword-spotting/keyword_dataset'

    keywords = ['backward', 'down', 'follow', 'forward',
                'go', 'left', 'off', 'on', 'right', 'stop']

    for keyword in keywords:
        path = os.path.join(DATASET_PATH, keyword)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Keyword directory not found: {path}")

    trainer = KeywordSpottingTrainer(
        dataset_path=DATASET_PATH,
        output_dir='./dnn_model'
    )

    train_loader, test_loader, label_encoder = trainer.prepare_data()
    print(f"Found {len(label_encoder.classes_)} classes:")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"{idx}: {label}")

    model = trainer.train(train_loader, test_loader, label_encoder, epochs=100)


if __name__ == '__main__':
    main()
