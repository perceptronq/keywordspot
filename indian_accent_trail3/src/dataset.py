import os
import torch
from torch.utils.data import Dataset
import torchaudio
import random

class CustomAudioDataset(Dataset):
    def __init__(self, dataset_path, labels=None, transform=None, split="all", test_size=0.2, random_state=42):
        """
        Custom dataset class for audio files.

        Args:
            dataset_path (str): Path to the dataset directory.
            labels (list or None): List of label names (subdirectories).
            transform (callable, optional): Transformation to apply to the audio data.
            split (str): "train", "test", or "all" to determine dataset split.
            test_size (float): Proportion of data to use for testing.
            random_state (int): Seed for reproducibility.
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.audio_paths = []
        self.labels = []

        if labels is None:
            # Auto-detect labels based on subdirectories
            labels = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

        label_to_idx = {label: idx for idx, label in enumerate(labels)}

        for label in labels:
            label_dir = os.path.join(dataset_path, label)
            for filename in os.listdir(label_dir):
                if filename.endswith(".wav"):
                    self.audio_paths.append(os.path.join(label_dir, filename))
                    self.labels.append(label_to_idx[label])

        # Shuffle and split dataset
        random.seed(random_state)
        combined = list(zip(self.audio_paths, self.labels))
        random.shuffle(combined)
        self.audio_paths, self.labels = zip(*combined)

        split_idx = int(len(self.audio_paths) * (1 - test_size))
        if split == "train":
            self.audio_paths = self.audio_paths[:split_idx]
            self.labels = self.labels[:split_idx]
        elif split == "test":
            self.audio_paths = self.audio_paths[split_idx:]
            self.labels = self.labels[split_idx:]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label
