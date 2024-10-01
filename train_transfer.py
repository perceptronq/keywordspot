import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchaudio.transforms import MFCC

# Configurations
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 2  # "cat" and "dog"
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 81  # Adjust according to your dataset (number of time steps in MFCC)

# Paths (replace with correct paths for your Kaggle environment)
DATA_DIR = '/kaggle/input/google-speech-commands'
BACKGROUND_NOISE_DIR = os.path.join(DATA_DIR, "_background_noise_")
TARGET_KEYWORDS = ["cat", "dog"]

# Custom Dataset
class SpeechCommandsDataset(Dataset):
    def __init__(self, data_dir, keywords, transform=None, augment_noise=False):
        self.data_dir = data_dir
        self.keywords = keywords
        self.transform = transform
        self.augment_noise = augment_noise
        self.audio_files = []
        self.noise_files = [os.path.join(BACKGROUND_NOISE_DIR, f) for f in os.listdir(BACKGROUND_NOISE_DIR) if f.endswith('.wav')]
        
        for keyword in self.keywords:
            keyword_path = os.path.join(self.data_dir, keyword)
            self.audio_files += [(os.path.join(keyword_path, f), keyword) for f in os.listdir(keyword_path) if f.endswith('.wav')]
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path, label = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.augment_noise:
            noise_waveform, _ = torchaudio.load(np.random.choice(self.noise_files))
            noise_waveform = noise_waveform[:, :waveform.size(1)]  # match length
            waveform += noise_waveform * 0.005  # small noise

        if self.transform:
            waveform = self.transform(waveform)

        waveform = self.pad_or_truncate(waveform)

        # Expand to 3 channels by duplicating across the channel dimension
        waveform = waveform.expand(3, -1, -1)

        label = 0 if label == "cat" else 1  # Label encoding
        return waveform, label


    def pad_or_truncate(self, mfcc):
        """Pads or truncates the MFCC tensor to the fixed length MAX_LEN."""
        if mfcc.size(2) < MAX_LEN:
            pad_size = MAX_LEN - mfcc.size(2)
            mfcc = torch.nn.functional.pad(mfcc, (0, pad_size), mode='constant', value=0)
        else:
            mfcc = mfcc[:, :, :MAX_LEN]
        return mfcc

# Data Augmentation and MFCC Transformation
mfcc_transform = MFCC(sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC)

# Dataset and Split
dataset = SpeechCommandsDataset(DATA_DIR, TARGET_KEYWORDS, transform=mfcc_transform, augment_noise=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model (Transfer Learning with MobileNetV2)
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)  # Modify final layer for binary classification

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

# Validation Loop
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return running_loss / len(loader.dataset), accuracy

# Training and Validation
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Save the model
torch.save(model.state_dict(), "keyword_spotting_mobilenetv2.pth")
