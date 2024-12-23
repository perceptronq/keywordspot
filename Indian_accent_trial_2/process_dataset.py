import os
import pickle
import librosa
import numpy as np

# Audio loading function
def load_audio(file_path, sr=16000):
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate

# MFCC feature extraction function
def extract_mfcc_features(file_path, sr=16000, n_mfcc=13):
    audio, _ = load_audio(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# Main function to process the dataset
def process_dataset(input_folder, output_file):
    features = []
    labels = []
    categories = ['check', 'up', 'left', 'background']
    
    for label, category in enumerate(categories):
        folder_path = os.path.join(input_folder, category)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            mfcc = extract_mfcc_features(file_path)
            features.append(mfcc)
            labels.append(label)  # Assign numeric label for each category
    
    # Save features and labels to a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump((features, labels), f)
    print(f"Saved features to {output_file}")

# Run feature extraction
process_dataset(input_folder="./dataset", output_file="./keyword_spotting_features.pkl")
