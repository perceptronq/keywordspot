import os
import librosa
import numpy as np
from hmmlearn import hmm
import pickle
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load audio and extract MFCC features
def extract_mfcc(file_path, sr=16000, n_mfcc=13):
    audio, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# Build dataset from the directory
def build_dataset(directory):
    features = []
    labels = []
    
    for label, category in enumerate(['check', 'background']):
        category_dir = os.path.join(directory, category)
        if not os.path.exists(category_dir):
            print(f"Directory '{category_dir}' not found.")
            continue

        for file_name in os.listdir(category_dir):
            file_path = os.path.join(category_dir, file_name)
            mfcc = extract_mfcc(file_path)
            features.append(mfcc)
            labels.append(label)  # 0 for "check", 1 for "background"
    
    return np.array(features), np.array(labels)

# Train GMM-HMM model
def train_gmmhmm(features, labels, n_components=3, n_iter=200):
    # Filter features for "check" class (label 0)
    check_features = features[labels == 0]

    # Reshape to (n_samples, n_features)
    check_features = check_features.reshape(-1, check_features.shape[1])

    # Initialize and train the model
    model_check = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=n_iter)
    model_check.fit(check_features)

    return model_check

# Predict using GMM-HMM model
def predict_gmmhmm(features, model_check):
    y_pred = []
    for mfcc in features:
        mfcc = mfcc.reshape(-1, len(mfcc))  # Reshape to 2D
        log_prob_check = model_check.score(mfcc)
        log_prob_background = -100  # Set a lower score for background class
        y_pred.append(0 if log_prob_check > log_prob_background else 1)
    return np.array(y_pred)

# Real-time recording and classification
def record_and_classify(model_check, duration=1, sr=16000, filename="test_audio.wav", use_existing_file=False):
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
    mfcc = mfcc.reshape(-1, len(mfcc))  # Reshape to 2D for HMM
    log_prob_check = model_check.score(mfcc)
    log_prob_background = -100  # Adjust if needed
    result = "check" if log_prob_check > log_prob_background else "background"
    print(f"Detected: {result}")
    return result

# Main function
def main():
    # Set dataset directory (relative to model_training directory)
    dataset_dir = '../keyword_spotting_dataset'
    
    # Build dataset
    print("Loading and extracting features from the dataset...")
    features, labels = build_dataset(dataset_dir)

    if len(features) == 0:
        print("No features found. Please check the dataset directory and try again.")
        return

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Split dataset into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train GMM-HMM model
    print("Training the GMM-HMM model...")
    model_check = train_gmmhmm(X_train, y_train, n_components=3, n_iter=200)

    # Save the trained model
    with open("hmmgmm_model.pkl", "wb") as file:
        pickle.dump(model_check, file)
    print("Model saved as 'hmmgmm_model.pkl'.")

    # Evaluate on training set
    print("Evaluating on training set...")
    y_train_pred = predict_gmmhmm(X_train, model_check)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.2f}")

    # Evaluate on validation set
    print("Evaluating on validation set...")
    y_valid_pred = predict_gmmhmm(X_valid, model_check)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    print(f"Validation Accuracy: {valid_accuracy:.2f}")

    # Test with real-time or existing audio
    print("Testing with audio...")
    use_existing_file = input("Do you want to use an existing file for classification? (yes/no): ").strip().lower() == "yes"

    if use_existing_file:
        filename = input("Enter the path to the audio file: ").strip()
        record_and_classify(model_check, use_existing_file=True, filename=filename)
    else:
        record_and_classify(model_check)

if __name__ == '__main__':
    main()
import sys
import pickle

def measure_gmmhmm_memory_usage(model_path, scaler_path):
    """
    Measures memory usage of a pickled GMM-HMM model and scaler.

    Parameters:
        model_path (str): Path to the pickled GMM-HMM model file.
        scaler_path (str): Path to the pickled scaler file.

    Returns:
        None: Prints memory usage of the model and scaler.
    """
    try:
        # Load the GMM-HMM model
        with open(model_path, "rb") as model_file:
            model_check = pickle.load(model_file)

        # Load the scaler
        with open(scaler_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        # Measure memory usage
        model_size = sys.getsizeof(model_check)
        scaler_size = sys.getsizeof(scaler)

        print(f"GMM-HMM Model Memory Usage: {model_size / 1024:.2f} KB")
        print(f"Scaler Memory Usage: {scaler_size / 1024:.2f} KB")
        print(f"Total Memory Usage: {(model_size + scaler_size) / 1024:.2f} KB")

    except Exception as e:
        print(f"Error measuring memory usage: {e}")

# Example Usage
if __name__ == '__main__':
    model_path = "hmmgmm_model.pkl"  # Path to the GMM-HMM model file
    scaler_path = "scaler.pkl"       # Path to the scaler file
    measure_gmmhmm_memory_usage(model_path, scaler_path)
