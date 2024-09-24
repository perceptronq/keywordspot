import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os

# Define the CNN model
class KeywordSpottingCNN(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(KeywordSpottingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # Calculate the size of the output from the last convolutional layer
        with torch.no_grad():
            x = torch.randn(1, 1, input_shape[0], input_shape[1])
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.relu3(self.conv3(x))
            fc1_input_size = x.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(fc1_input_size, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom dataset class
class SpeechCommandsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.data = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for audio_file in os.listdir(class_dir):
                if audio_file.endswith('.wav'):
                    self.data.append((os.path.join(class_dir, audio_file), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, label = self.data[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label

# Audio preprocessing
def preprocess_audio(waveform):
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to 16kHz if needed
    if waveform.shape[1] != 16000:
        waveform = torchaudio.transforms.Resample(waveform.shape[1], 16000)(waveform)
    
    # Pad or truncate to 1 second (16000 samples)
    if waveform.shape[1] < 16000:
        waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
    elif waveform.shape[1] > 16000:
        waveform = waveform[:, :16000]
    
    # Convert to mel spectrogram
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=40)(waveform)
    
    # Convert to decibels
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    return mel_spec

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Main execution
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    dataset = SpeechCommandsDataset('/home/mayankch283/Downloads/speech_commands_v0.02', transform=preprocess_audio)
    
    # Print class information
    print(f"Number of classes: {len(dataset.classes)}")
    print("Classes:")
    for i, class_name in enumerate(dataset.classes):
        print(f"  {i}: {class_name}")
    
    # Hyperparameters
    num_classes = 35  # Adjust based on your dataset
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # Create dataset and dataloader
    dataset = SpeechCommandsDataset('/home/mayankch283/Downloads/speech_commands_v0.02', transform=preprocess_audio)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get the shape of the preprocessed audio
    sample_data, _ = dataset[0]
    input_shape = sample_data.shape[1:]  # Exclude batch dimension

    # Create model, loss function, and optimizer
    model = KeywordSpottingCNN(num_classes, input_shape).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch+1} completed')

    print("Training finished")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()