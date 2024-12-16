import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MFCC
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader, random_split
import random
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 35  # Number of keywords in the dataset

class SpeechCommandsDataset(SPEECHCOMMANDS):
    def __init__(self, subset):
        super().__init__(".", download=True, subset=subset)
        self.mfcc_transform = MFCC(n_mfcc=40, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40})
        self.noise_files = [f for f in self._walker if "background_noise" in f]
        

    def __getitem__(self, n):
        waveform, sample_rate, label, _, _ = super().__getitem__(n)
        
        # Convert to MFCC
        mfcc = self.mfcc_transform(waveform)

        # Data augmentation: Add noise
        if random.random() < 0.5:
            noise_file = random.choice(self.noise_files)
            noise, _ = torchaudio.load(noise_file)
            noise = noise[:, :waveform.shape[1]]  # Trim noise to match waveform length
            noise_level = random.uniform(0, 0.1)
            augmented_waveform = waveform + noise_level * noise
            mfcc = self.mfcc_transform(augmented_waveform)

        # Pad or trim to fixed length
        if mfcc.shape[2] < 101:
            mfcc = nn.functional.pad(mfcc, (0, 101 - mfcc.shape[2]))
        else:
            mfcc = mfcc[:, :, :101]

        if noise.shape[1] < waveform.shape[1]:
            noise = nn.functional.pad(noise, (0, waveform.shape[1] - noise.shape[1]))
        noise = noise[:, :waveform.shape[1]]

        return mfcc, self._label_to_index[label]

# Load and preprocess the data
train_dataset = SpeechCommandsDataset("training")
val_dataset = SpeechCommandsDataset("validation")
test_dataset = SpeechCommandsDataset("testing")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Define the model
class KeywordSpottingModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mobilenet = mobilenet_v2(pretrained=True)
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

model = KeywordSpottingModel(NUM_CLASSES)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    train_accuracy = train_correct / train_total
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_accuracy = val_correct / val_total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Test the model
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_accuracy = test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model
torch.save(model.state_dict(), "keyword_spotting_model.pth")
print("Model saved as keyword_spotting_model.pth")