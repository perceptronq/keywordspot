import os
import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.preprocessing import StandardScaler
import pickle

# Load audio and extract MFCC features
def extract_mfcc(file_path, sr=16000, n_mfcc=13):
    audio, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# Function to record audio or classify an existing file
def record_and_classify(model, scaler, duration=1, sr=16000, filename="test_audio.wav", use_existing_file=False):
    """Records audio if `use_existing_file` is False; otherwise, uses an existing file for classification."""
    
    if not use_existing_file:
        input("Press Enter to start recording...")  # Wait for user to get ready
        print("Recording... please say 'check' if that's the keyword.")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        write(filename, sr, audio)  # Save as WAV file
        print("Recording complete.")
    else:
        print(f"Using existing audio file: {filename}")

    # Extract MFCC and classify
    mfcc = extract_mfcc(filename, sr=sr)
    mfcc_scaled = scaler.transform([mfcc])  # Scale MFCC features
    result = model.predict(mfcc_scaled)
    label = "check" if result == 0 else "background"
    print(f"Detected: {label}")
    return label

# Main function
def main():
    # Load the trained SVM model and scaler
    with open("svm_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    print("Loaded SVM model.")

    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Loaded scaler.")

    # Test with real-time or existing audio
    print("Testing with audio...")
    use_existing_file = input("Do you want to use an existing file for classification? (yes/no): ").strip().lower() == "yes"

    if use_existing_file:
        filename = input("Enter the path to the audio file: ").strip()
        record_and_classify(model, scaler, use_existing_file=True, filename=filename)
    else:
        record_and_classify(model, scaler)

if __name__ == '__main__':
    main()

import pickle
import sys

def measure_model_memory_usage(model_path, scaler_path):
    """
    Measures memory usage of a pickled model and scaler.
    
    Parameters:
        model_path (str): Path to the pickled model file.
        scaler_path (str): Path to the pickled scaler file.
    
    Returns:
        None: Prints memory usage of the model and scaler.
    """
    try:
        # Load the model and scaler
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        
        with open(scaler_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        # Measure memory usage
        model_size = sys.getsizeof(model)
        scaler_size = sys.getsizeof(scaler)

        print(f"Model Memory Usage: {model_size / 1024:.2f} KB")
        print(f"Scaler Memory Usage: {scaler_size / 1024:.2f} KB")
        print(f"Total Memory Usage: {(model_size + scaler_size) / 1024:.2f} KB")

    except Exception as e:
        print(f"Error measuring memory usage: {e}")

# Example Usage
if __name__ == '__main__':
    model_path = "svm_model.pkl"  # Path to the SVM model file
    scaler_path = "scaler.pkl"   # Path to the scaler file
    measure_model_memory_usage(model_path, scaler_path)
