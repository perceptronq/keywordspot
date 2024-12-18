import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels, transform=None):
        self.audio_paths = audio_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.audio_paths[idx])

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        if waveform.shape[1] > 16000:
            waveform = waveform[:, :16000]
        else:
            padding = 16000 - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        spectrogram_transformer = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=64,
            n_fft=1024,
            hop_length=512
        )
        spectrogram = spectrogram_transformer(waveform)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, self.labels[idx]

class AudioAugmentation:
    @staticmethod
    def add_noise(audio, noise_factor=0.005):
        noise = torch.randn_like(audio) * noise_factor
        return audio + noise

    @staticmethod
    def time_shift(audio, shift_limit=0.1):
        shift = int(np.random.uniform(-shift_limit,
                    shift_limit) * audio.shape[-1])
        return torch.roll(audio, shift, dims=-1)

    @staticmethod
    def time_mask(audio, mask_param=50, p=0.5):
        if torch.rand(1).item() < p:
            max_mask_width = min(mask_param, audio.shape[-1])
            if max_mask_width < 1:
                return audio
            mask_width = torch.randint(1, max_mask_width + 1, (1,)).item()
            start_limit = audio.shape[-1] - mask_width
            if start_limit > 0:
                start = torch.randint(0, start_limit + 1, (1,)).item()
            else:
                start = 0
            audio[..., start:start+mask_width] = 0
        return audio

    @staticmethod
    def mix_background(audio, background, mix_factor=0.1):
        if len(background) >= len(audio):
            background = background[:len(audio)]
        else:
            padding = len(audio) - len(background)
            background = torch.nn.functional.pad(background, (0, padding))
        return (1 - mix_factor) * audio + mix_factor * background

class AddNoise(nn.Module):
    def __init__(self, noise_factor=0.005):
        super().__init__()
        self.noise_factor = noise_factor

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise

class TimeMask(nn.Module):
    def __init__(self, mask_param=50, p=0.5):
        super().__init__()
        self.mask_param = mask_param
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() < self.p:
            max_mask_width = min(self.mask_param, x.shape[-1])
            if max_mask_width < 1:
                return x
            mask_width = torch.randint(1, max_mask_width + 1, (1,)).item()
            start_limit = x.shape[-1] - mask_width
            if start_limit > 0:
                start = torch.randint(0, start_limit + 1, (1,)).item()
            else:
                start = 0
            x = x.clone()
            x[..., start:start+mask_width] = 0
        return x

class KeywordSpottingCNN(nn.Module):
    def __init__(self, num_classes):
        super(KeywordSpottingCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class KeywordSpottingTrainer:
    def __init__(self, dataset_path, output_dir='./cnn_model_v1'):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data(self, test_size=0.2, random_state=42, batch_size=32):
        audio_paths = []
        labels = []

        for keyword_folder in os.listdir(self.dataset_path):
            keyword_path = os.path.join(self.dataset_path, keyword_folder)
            if os.path.isdir(keyword_path):
                for filename in os.listdir(keyword_path):
                    if filename.endswith('.wav'):
                        file_path = os.path.join(keyword_path, filename)
                        audio_paths.append(file_path)
                        labels.append(keyword_folder)

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            audio_paths, encoded_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=encoded_labels
        )

        train_transform = nn.Sequential(
            AddNoise(),
            TimeMask()
        )

        train_dataset = AudioDataset(
            X_train, y_train, transform=train_transform)
        test_dataset = AudioDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, label_encoder

    def train(self, train_loader, test_loader, label_encoder, epochs=50):
        model = KeywordSpottingCNN(num_classes=len(
            label_encoder.classes_)).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5)

        train_losses = []
        test_losses = []
        accuracies = []

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for spectrograms, labels in train_loader:
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)

                outputs = model(spectrograms)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for spectrograms, labels in test_loader:
                    spectrograms = spectrograms.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(spectrograms)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            train_loss /= len(train_loader)
            test_loss /= len(test_loader)
            accuracy = 100 * correct / total

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            scheduler.step(test_loss)

            print(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}, "
                f"Accuracy: {accuracy:.2f}%"
            )

        torch.save(model.state_dict(), os.path.join(
            self.output_dir, 'cnn_model.pth'))

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 3, 2)
        plt.plot(test_losses, label='Test Loss', color='red')
        plt.title('Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 3, 3)
        plt.plot(accuracies, label='Accuracy', color='green')
        plt.title('Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'learning_curves.png'))

        return model, train_losses, test_losses

    def evaluate(self, model, test_loader, label_encoder):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for spectrograms, labels in test_loader:
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)

                outputs = model(spectrograms)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds,
                                    target_names=label_encoder.classes_))

def main():
    DATASET_PATH = './keyword_dataset'

    trainer = KeywordSpottingTrainer(DATASET_PATH)

    train_loader, test_loader, label_encoder = trainer.prepare_data()

    model, train_losses, test_losses = trainer.train(
        train_loader, test_loader, label_encoder
    )

    trainer.evaluate(model, test_loader, label_encoder)


if __name__ == '__main__':
    main()
