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

        waveform, sample_rate = torchaudio.load(filepath)
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

# Data Preprocessing and MFCC Transform
def get_dataset(data_dir, labels):
    mfcc_transform = MFCC(
        sample_rate=16000, n_mfcc=13, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
    )
    dataset = KeywordSpottingDataset(data_dir, labels, transform=mfcc_transform)
    return dataset

# Custom collate function for padding to a fixed length
def collate_fn(batch, fixed_length=215):
    waveforms = []
    labels = []
    for waveform, label in batch:
        # Pad or truncate the waveform to the fixed length
        if waveform.size(2) < fixed_length:
            padding = torch.zeros(1, waveform.size(1), fixed_length - waveform.size(2))  # Match dim=1 (13)
            waveform = torch.cat((waveform, padding), dim=2)
        else:
            waveform = waveform[:, :, :fixed_length]  # Truncate if too long
        waveforms.append(waveform)
        labels.append(label)

    # Stack all tensors to form a batch
    waveforms = torch.stack(waveforms)
    labels = torch.tensor(labels)

    return waveforms, labels
# CNN Model Definition
class SimpleCNN(nn.Module):
    def __init__(self, n_input=13, n_output=4, fixed_length=215):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2)
        
        # Calculate the output size after convolutions and pooling
        sample_input = torch.zeros(1, 1, n_input, fixed_length)
        sample_output = self.pool(F.relu(self.conv1(sample_input)))
        sample_output = self.pool(F.relu(self.conv2(sample_output)))
        flattened_size = sample_output.numel()  # Total number of elements after flattening

        # Define fully connected layers with correct flattened input size
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, n_output)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Training Function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        # Debug: Print the shape of inputs before feeding to model
        print("Shape of inputs before model in train_epoch:", inputs.shape)

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
            inputs = inputs.unsqueeze(1)  # Add channel dimension

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
    model = SimpleCNN().to(device)
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
    torch.save(model.state_dict(), "keyword_spotting_model.pth")
    print("Model saved as keyword_spotting_model.pth")

if __name__ == "__main__":
    main()
