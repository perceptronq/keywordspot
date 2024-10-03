import torch
import torchaudio
import pyaudio
import numpy as np
from torchvision import models
from torchaudio.transforms import MFCC
import torch.nn as nn

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 5)  # Update the classifier to 5 classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("keyword_spotting_mobilenetv2.pth", map_location=device))
model.to(device)
model.eval()

SAMPLE_RATE = 16000
N_MFCC = 40  
MAX_LEN = 81
CHUNK = int(SAMPLE_RATE * 0.5)

mfcc_transform = MFCC(sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening for keywords: cat, dog, bird, yes, no...")

def preprocess_audio(audio_data):
    """Preprocess raw audio data into MFCC features"""
    audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)  
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
        mfcc = mfcc.to(device)  # Move input to the same device as the model
        outputs = model(mfcc)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    return predicted.item(), probabilities[0]

KEYWORDS = ['cat', 'dog', 'bird', 'yes', 'no']

try:
    while True:
        audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
        mfcc = preprocess_audio(audio_data)        
        prediction, probabilities = predict_keyword(mfcc)
        confidence_threshold = 0.7
        if probabilities[prediction].item() > confidence_threshold:
            detected_keyword = KEYWORDS[prediction]
            confidence = probabilities[prediction].item()
            print(f"Detected: {detected_keyword} (Confidence: {confidence:.2f})")

except KeyboardInterrupt:
    print("Stopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()