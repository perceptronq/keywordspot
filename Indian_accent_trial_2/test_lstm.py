import os
import torch
import torch.nn as nn  # This line is missing
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MFCC


# Define labels (same order as during training)
LABELS = ["check", "left", "up", "background"]

# Model Definition (must match the trained model)
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
        return torch.softmax(x, dim=1)

# Load the model
model = KeywordSpottingLSTM(n_mfcc=13, n_hidden=128, n_output=len(LABELS))
model.load_state_dict(torch.load("keyword_spotting_lstm_model.pth"))
model.eval()

# Preprocessing Function
def preprocess_audio(filepath, fixed_length=215):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(filepath)

    # Convert to mono if stereo
    if waveform.size(0) > 1:  # Stereo to Mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Apply MFCC transformation
    mfcc_transform = MFCC(
        sample_rate=16000,
        n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
    )
    waveform = mfcc_transform(waveform)

    # Pad or truncate to fixed length
    if waveform.size(2) < fixed_length:
        padding = torch.zeros(1, waveform.size(1), fixed_length - waveform.size(2))
        waveform = torch.cat((waveform, padding), dim=2)
    else:
        waveform = waveform[:, :, :fixed_length]

    # Reshape for LSTM input (time_steps, n_mfcc)
    waveform = waveform.squeeze(0).transpose(0, 1)  # Shape: [time_steps, n_mfcc]

    return waveform.unsqueeze(0)  # Add batch dimension: [1, time_steps, n_mfcc]

# Prediction Function
def predict(filepath):
    # Preprocess the audio
    input_tensor = preprocess_audio(filepath).to(torch.device("cpu"))

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()

    # Get the corresponding label name
    return LABELS[predicted_label]

# Interactive Test Function
def test_model():
    filepath = input("Enter the path to the audio file (.wav): ")
    if not filepath.lower().endswith(".wav"):
        print("Error: Please provide a .wav audio file.")
        return

    if not os.path.exists(filepath):
        print("Error: File not found. Please check the path and try again.")
        return

    # Run prediction
    predicted_label = predict(filepath)
    print(f"Predicted Label: {predicted_label}")

# Run the test
if __name__ == "__main__":
    test_model()
