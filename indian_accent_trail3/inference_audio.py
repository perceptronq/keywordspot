import time
import torch
import torchaudio
import argparse
from src.model import KeywordSpottingModel
from src.config import load_config
import os
import pickle
import torchaudio.transforms as transforms
import sounddevice as sd
import numpy as np

def load_model(config):
    device = torch.device("cpu")
    model = KeywordSpottingModel(num_classes=config["num_classes"])
    model.load_state_dict(torch.load(config["model_save_path"], map_location=device))
    model.eval()
    return model, device

def load_label_encoder(config):
    with open(config["label_encoder_save_path"], "rb") as f:
        label_encoder = pickle.load(f)
    return label_encoder

def preprocess_audio(audio_path, sample_rate=16000, duration=1.0):
    waveform, orig_sample_rate = torchaudio.load(audio_path)
    if orig_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    target_length = int(sample_rate * duration)
    if waveform.shape[1] < target_length:
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :target_length]
    mel_transform = transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=64
    )
    spectrogram = mel_transform(waveform).unsqueeze(0)
    return spectrogram

def predict_word(model, device, label_encoder, audio_path):
    spectrogram = preprocess_audio(audio_path).to(device)
    with torch.no_grad():
        output = model(spectrogram)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = probabilities.max(dim=1)
    if confidence.item() < 0.6:
        return "Background Noise or Unknown"
    predicted_word = label_encoder.inverse_transform([predicted_idx.item()])[0]
    return predicted_word

def predict_from_microphone(model, device, label_encoder, duration=1.0, sample_rate=16000):
    # Notify user of upcoming recording
    print("Get ready to record!")
    for i in range(3, 0, -1):  # 3-second countdown
        print(f"Recording starts in {i}...")
        time.sleep(1)

    print("Recording now!")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording stopped.")

    # Convert to waveform tensor
    waveform = torch.tensor(recording.T).float()
    mel_transform = transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=64
    )
    spectrogram = mel_transform(waveform).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(spectrogram)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = probabilities.max(dim=1)
    if confidence.item() < 0.6:
        return "Background Noise or Unknown"
    predicted_word = label_encoder.inverse_transform([predicted_idx.item()])[0]
    return predicted_word

def main():
    config = load_config()
    model, device = load_model(config)
    label_encoder = load_label_encoder(config)
    
    print("Do you want to:")
    print("1. Input an audio file")
    print("2. Use your microphone to speak")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        audio_path = input("Enter the path to the audio file: ").strip()
        if not os.path.exists(audio_path):
            print(f"Error: Audio file '{audio_path}' does not exist.")
            return
        predicted_word = predict_word(model, device, label_encoder, audio_path)
        print(f"Predicted Word: {predicted_word}")
    elif choice == "2":
        try:
            duration = float(input("Enter recording duration in seconds (default: 1): ").strip() or 1.0)
        except ValueError:
            duration = 1.0
        predicted_word = predict_from_microphone(model, device, label_encoder, duration=duration)
        print(f"Predicted Word: {predicted_word}")
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
