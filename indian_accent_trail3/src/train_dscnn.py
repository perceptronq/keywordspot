import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.dataset import CustomAudioDataset
from src.dscnn_model import DSCNN
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt

def train_dscnn(config):
    # Check if dataset path exists
    if not os.path.exists(config["dataset_path"]):
        print(f"Error: Dataset path '{config['dataset_path']}' does not exist.")
        return

    # Load dataset
    audio_paths = []
    labels = []
    print("Loading dataset...")
    for keyword in os.listdir(config["dataset_path"]):
        keyword_path = os.path.join(config["dataset_path"], keyword)
        if os.path.isdir(keyword_path):
            for file in os.listdir(keyword_path):
                if file.endswith('.wav'):
                    audio_paths.append(os.path.join(keyword_path, file))
                    labels.append(keyword)

    if not audio_paths:
        print("Error: No audio files found in the dataset path.")
        return
    else:
        print(f"Total audio files found: {len(audio_paths)}")

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Train-test split
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        audio_paths, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    print(f"Dataset split into {len(train_paths)} training samples and {len(test_paths)} test samples.")

    # Transform for spectrograms
    mel_transform = transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64
    )

    # Create datasets and loaders
    train_dataset = CustomAudioDataset(train_paths, train_labels, transform=mel_transform)
    test_dataset = CustomAudioDataset(test_paths, test_labels, transform=mel_transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    # Initialize model
    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DSCNN(num_classes=len(label_encoder.classes_))
    model.to(device)
    print("Model initialized successfully.")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    epochs = config["epochs"]

    train_losses = []
    test_accuracies = []

    print("Starting training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        running_loss = 0.0

        for i, (spectrograms, labels) in enumerate(train_loader):
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Log every 10 batches
            if (i + 1) % 10 == 0:
                print(f"  Batch {i + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # Calculate training loss for the epoch
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        print(f"  Train Loss: {train_loss:.4f}")

        # Evaluate on the test set
        print("  Evaluating on test set...")
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for spectrograms, labels in test_loader:
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                outputs = model(spectrograms)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Calculate accuracy
        accuracy = 100.0 * correct / total
        test_accuracies.append(accuracy)
        print(f"  Test Accuracy: {accuracy:.2f}%")

    # Save model and encoder
    os.makedirs(os.path.dirname(config["dscnn_model_save_path"]), exist_ok=True)
    torch.save(model.state_dict(), config["dscnn_model_save_path"])

    with open(config["label_encoder_save_path"], "wb") as f:
        import pickle
        pickle.dump(label_encoder, f)

    print("Training completed. Model and artifacts saved.")

    # Plot training loss
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(config["output_path"], "training_loss_dscnn.png"))

    print("Training loss curve saved to outputs folder.")
