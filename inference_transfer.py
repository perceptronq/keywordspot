import torch
import torchaudio
import pyaudio
import numpy as np
from torchvision import models
from torchaudio.transforms import MFCC

import numpy as np
import torch.nn as nn

# Load the trained model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # Update the classifier to 2 classes
model.load_state_dict(torch.load("keyword_spotting_mobilenetv2.pth"))
model.eval()

# Audio settings
SAMPLE_RATE = 16000  # Must match the sample rate used in training
N_MFCC = 40          # Must match the number of MFCC coefficients
MAX_LEN = 81         # Must match the input length to the model
CHUNK = int(SAMPLE_RATE * 1)  # 1-second chunks for real-time processing

# MFCC transformation (same as during training)
mfcc_transform = MFCC(sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC)

# Initialize pyaudio to capture audio from the microphone
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening...")

def preprocess_audio(audio_data):
    """Preprocess raw audio data into MFCC features"""
    audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)  # Add batch dimension
    mfcc = mfcc_transform(audio_tensor)
    # Pad or truncate to match the input length for the model
    if mfcc.size(2) < MAX_LEN:
        pad_size = MAX_LEN - mfcc.size(2)
        mfcc = torch.nn.functional.pad(mfcc, (0, pad_size), mode='constant', value=0)
    else:
        mfcc = mfcc[:, :, :MAX_LEN]
    
    # Expand to 3 channels (for MobileNetV2 input)
    mfcc = mfcc.expand(3, -1, -1)
    return mfcc.unsqueeze(0)  # Add batch dimension

def predict_keyword(mfcc):
    """Run inference on the model to predict the keyword"""
    with torch.no_grad():
        outputs = model(mfcc)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

try:
    while True:
        # Capture audio
        audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        
        # Preprocess audio data into MFCC features
        mfcc = preprocess_audio(audio_data)
        
        # Run inference to predict keyword
        prediction = predict_keyword(mfcc)
        
        if prediction == 0:
            print("Detected: Cat")
        elif prediction == 1:
            print("Detected: Dog")

except KeyboardInterrupt:
    # Stop the stream and close pyaudio
    print("Stopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()
