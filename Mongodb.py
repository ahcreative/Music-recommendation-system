import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pymongo import MongoClient
from tqdm import tqdm
import librosa
import zipfile

# Function to read metadata files from a zip archive
def read_metadata_files_from_zip(metadata_zip_path):
    with zipfile.ZipFile(metadata_zip_path, 'r') as zip_ref:
        zip_ref.extractall('metadata_temp')  # Extract all contents to a temporary directory
    metadata_dir = 'metadata_temp'  # Temporary directory where metadata files are extracted
    albums = pd.read_csv(os.path.join(metadata_dir, 'raw_albums.csv'))
    artists = pd.read_csv(os.path.join(metadata_dir, 'raw_artists.csv'))
    genres = pd.read_csv(os.path.join(metadata_dir, 'raw_genres.csv'))
    tracks = pd.read_csv(os.path.join(metadata_dir, 'raw_tracks.csv'))
    # Clean up temporary directory
    os.rmdir(metadata_dir)  # Remove temporary directory
    return albums, artists, genres, tracks

# Function to read metadata files
def read_metadata_files(metadata_dir):
    albums = pd.read_csv(os.path.join(metadata_dir, 'raw_albums.csv'))
    artists = pd.read_csv(os.path.join(metadata_dir, 'raw_artists.csv'))
    genres = pd.read_csv(os.path.join(metadata_dir, 'raw_genres.csv'))
    tracks = pd.read_csv(os.path.join(metadata_dir, 'raw_tracks.csv'))
    return albums, artists, genres, tracks

# Function to preprocess metadata
def preprocess_metadata(albums, artists, genres, tracks):
    # Merge metadata based on track_id
    metadata = pd.merge(tracks, albums, on='album_id', how='left')
    metadata = pd.merge(metadata, artists, on='artist_id', how='left')
    metadata = pd.merge(metadata, genres, on='track_id', how='left')
    return metadata

def extract_audio_paths(audio_dir):
    audio_paths = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".mp3"):
                path = os.path.join(root, file)
                audio_paths.append(path)
    return audio_paths

def extract_features(audio_file):
    try:
        # Try to load audio file using librosa
        y, sr = librosa.load(audio_file, sr=None)
    except Exception as e:
        # If loading fails, print a warning and return None
        print(f"Warning: Failed to load audio file '{audio_file}': {e}")
        return None
    
    # Proceed with feature extraction if loading is successful
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]

    # Aggregate statistics over each feature
    features = []
    for feature in [mfcc, spectral_centroid, zero_crossing_rate]:
        features.append(np.mean(feature))
        features.append(np.std(feature))
        features.append(np.median(feature))
    return features

def preprocess_data(features):
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Perform dimensionality reduction using PCA
    pca = PCA(n_components=5)
    reduced_features = pca.fit_transform(scaled_features)
    
    return reduced_features

def load_data_to_mongodb(features, metadata):
    # Connect to MongoDB
    client = MongoClient('192.168.93.129', 27017)
    db = client['music_features']
    collection = db['audio_features']
    
    # Insert features into MongoDB
    for i, (feature, row) in tqdm(enumerate(zip(features, metadata.iterrows())), total=len(features), desc="Loading data into MongoDB"):
        track_id = row['track_id']
        document = {
            "track_id": track_id,
            "features": feature.tolist(),
            "metadata": row.to_dict()
        }
        collection.insert_one(document)

def main():
    # Step 1: Extract audio paths
    audio_dir = r'/home/laiba/Documents/sample_size'
    audio_paths = extract_audio_paths(audio_dir)
    
    if len(audio_paths) == 0:
        print("No audio files found in the directory.")
        return
    
    # Step 2: Extract features
    features = []
    for audio_file in audio_paths:
        feature = extract_features(audio_file)
        if feature is not None:
            features.append(feature)
    features = np.array(features)
    
    if len(features) == 0:
        print("No features extracted.")
        return
    
    # Step 3: Preprocessing
    scaled_features = preprocess_data(features)
    
    # Step 4: Read and preprocess metadata
    metadata_dir = r'/home/laiba/fma_metadata.zip'
    albums, artists, genres, tracks = read_metadata_files(metadata_dir)
    metadata = preprocess_metadata(albums, artists, genres, tracks)
    
    # Step 5: Load data into MongoDB
    load_data_to_mongodb(scaled_features, metadata)

if _name_ == "main":
    main()
