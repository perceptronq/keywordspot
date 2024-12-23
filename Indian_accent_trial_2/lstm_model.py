import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.transforms import MFCC
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset Class
class KeywordSpottingDataset(Dataset):
    def __init__(self, data_dir, labels, transform=None):
        self.data_dir = data_dir
        self.labels = labels
        self.transform = transform
        self.filepaths = []
        self.label_indices = []

        for label, folder in enumerate(labels):
            folder_path = os.path.join(data_dir, folder)
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    self.filepaths.append(os.path.join(folder_path, file))
                    self.label_indices.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.label_indices[idx]

        # Load audio file
        waveform, sample_rate = torchaudio.load(filepath)

        # Convert to mono if stereo
        if waveform.size(0) > 1:  # More than one channel
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply transform if provided
        if self.transform:
            waveform = self.transform(waveform)

        # Debug: Print shape after conversion and transformation
        if waveform.ndim != 3 or waveform.size(1) != 13:
            raise ValueError(f"Unexpected waveform shape: {waveform.shape}. Expected [1, 13, time_steps]")

        return waveform, label

# Data Preprocessing and MFCC Transform
def get_dataset(data_dir, labels):
    mfcc_transform = MFCC(
        sample_rate=16000, 
        n_mfcc=13, 
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
    )
    dataset = KeywordSpottingDataset(data_dir, labels, transform=mfcc_transform)
    return dataset

# Custom collate function for padding/truncation
def collate_fn(batch, fixed_length=215):
    waveforms = []
    labels = []
    for waveform, label in batch:
        # Debug: Print shape before processing
        print(f"Original waveform shape: {waveform.shape}")

        # Ensure waveform has three dimensions
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.size(1) != 13:
            raise ValueError(f"Unexpected MFCC dimension: {waveform.size(1)}. Expected 13.")

        # Pad or truncate the waveform to the fixed length
        if waveform.size(2) < fixed_length:
            padding = torch.zeros(1, waveform.size(1), fixed_length - waveform.size(2))
            waveform = torch.cat((waveform, padding), dim=2)
        else:
            waveform = waveform[:, :, :fixed_length]

        waveforms.append(waveform.squeeze(0).transpose(0, 1))  # Shape: [time_steps, n_mfcc]
        labels.append(label)

    waveforms = torch.stack(waveforms)  # Shape: [batch_size, time_steps, n_mfcc]
    labels = torch.tensor(labels)

    # Debug: Print final batch shape
    print(f"Final batch shape: {waveforms.shape}")

    return waveforms, labels

# LSTM Model Definition
class KeywordSpottingLSTM(nn.Module):
    def __init__(self, n_mfcc=13, n_hidden=128, n_output=4):
        super(KeywordSpottingLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_mfcc, hidden_size=n_hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # x shape: [batch_size, time_steps, n_mfcc]
        x, _ = self.lstm(x)  # LSTM outputs hidden states
        x = x[:, -1, :]  # Take the last hidden state
        x = self.fc(x)  # Fully connected layer
        return F.log_softmax(x, dim=1)

# Training Function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    accuracy = 100 * correct / total
    return total_loss / len(train_loader), accuracy

# Validation Function
def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    accuracy = 100 * correct / total
    return total_loss / len(val_loader), accuracy

# Main Training Script
def main():
    # Parameters
    data_dir = "./dataset"  # Path to your dataset
    labels = ["check", "left", "up", "background"]
    batch_size = 32
    n_epochs = 20
    learning_rate = 0.001
    fixed_length = 215  # Fixed length for padding/truncation

    # Dataset and DataLoader
    dataset = get_dataset(data_dir, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, fixed_length))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, fixed_length))

    # Model, Loss, Optimizer
    model = KeywordSpottingLSTM(n_mfcc=13, n_hidden=128, n_output=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    # Save the Model
    torch.save(model.state_dict(), "keyword_spotting_lstm_model.pth")
    print("Model saved as keyword_spotting_lstm_model.pth")

if __name__ == "__main__":
    main()
