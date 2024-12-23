import torch
import torchaudio
from src.model import KeywordSpottingModel
from src.config import load_config
import pickle
import torchaudio.transforms as transforms

def run_inference(config, audio_path):
    # Load model
    device = torch.device("cpu")
    model = KeywordSpottingModel(num_classes=config["num_classes"])
    model.load_state_dict(torch.load(config["model_save_path"], map_location=device))
    model.eval()

    # Load label encoder
    with open(config["label_encoder_save_path"], "rb") as f:
        label_encoder = pickle.load(f)

    # Preprocess audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel_transform = transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64
    )
    spectrogram = mel_transform(waveform).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(spectrogram)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = probabilities.max(dim=1)

    # Decode label
    if confidence.item() < 0.6:
        print("Prediction: Background Noise or Unknown")
    else:
        predicted_word = label_encoder.inverse_transform([predicted_idx.item()])[0]
        print(f"Prediction: {predicted_word} (Confidence: {confidence.item():.2f})")
