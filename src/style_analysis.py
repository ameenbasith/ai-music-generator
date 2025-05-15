import os
import json
import numpy as np
import pandas as pd
import librosa
from src.audio_processing import AudioProcessor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class StyleAnalyzer:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.features_cache = {}
        self.style_vectors = {}

    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(i) for i in obj]
        else:
            return obj

    def extract_features_from_directory(self, directory, artist_name):
        """Extract features from all audio files in a directory"""
        features_list = []

        # Get all wav and mp3 files
        audio_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.wav', '.mp3')):
                    audio_files.append(os.path.join(root, file))

        print(f"Found {len(audio_files)} audio files for {artist_name}")

        # Process each file
        for audio_file in audio_files:
            print(f"Processing {os.path.basename(audio_file)}")
            try:
                audio, sr = self.audio_processor.load_audio(audio_file)
                features = self.audio_processor.extract_features(audio, sr)
                features['file'] = os.path.basename(audio_file)
                features_list.append(features)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

        # Cache the features
        self.features_cache[artist_name] = features_list

        # Save features to disk
        output_path = f"data/processed_audio/{artist_name}_features.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Ensure features are JSON serializable
        serializable_features = self._make_json_serializable(features_list)

        with open(output_path, 'w') as f:
            json.dump(serializable_features, f)

        print(f"Saved features for {artist_name} to {output_path}")

        return features_list

    def create_style_vector(self, artist_name):
        """Create a style vector for an artist based on their features"""
        if artist_name not in self.features_cache:
            print(f"No features found for {artist_name}. Extract features first.")
            return None

        features = self.features_cache[artist_name]

        # Convert to DataFrame for easier processing
        df_list = []
        for feature_dict in features:
            # Create a flat dict without nested lists
            flat_dict = {
                'file': feature_dict['file'],
                'spectral_centroid_mean': feature_dict['spectral_centroid_mean'],
                'spectral_bandwidth_mean': feature_dict['spectral_bandwidth_mean'],
                'spectral_rolloff_mean': feature_dict['spectral_rolloff_mean'],
                'tempo': feature_dict['tempo']
            }

            # Add chroma features
            for i, val in enumerate(feature_dict['chroma_means']):
                flat_dict[f'chroma_{i}'] = val

            # Add MFCC features
            for i, val in enumerate(feature_dict['mfcc_means']):
                flat_dict[f'mfcc_{i}'] = val

            df_list.append(flat_dict)

        df = pd.DataFrame(df_list)

        # Drop non-numeric columns for analysis
        analysis_df = df.drop('file', axis=1)

        # Normalize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(analysis_df)

        # Create a style vector (mean of all features)
        style_vector = np.mean(scaled_data, axis=0)

        # Store the style vector
        self.style_vectors[artist_name] = {
            'vector': style_vector.tolist(),  # Convert to list for JSON
            'feature_names': analysis_df.columns.tolist()
        }

        # Save style vector to disk
        output_path = f"data/processed_audio/{artist_name}_style_vector.json"

        # Ensure the vector is JSON serializable
        serializable_vector = self._make_json_serializable(self.style_vectors[artist_name])

        with open(output_path, 'w') as f:
            json.dump(serializable_vector, f)

        print(f"Created style vector for {artist_name}")
        return style_vector

    def compare_styles(self, artist1, artist2):
        """Compare two artists' styles and return similarity score"""
        if artist1 not in self.style_vectors or artist2 not in self.style_vectors:
            print("Both artists must have style vectors extracted first")
            return None

        vector1 = np.array(self.style_vectors[artist1]['vector'])
        vector2 = np.array(self.style_vectors[artist2]['vector'])

        # Cosine similarity
        similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

        print(f"Similarity between {artist1} and {artist2}: {similarity:.2f}")
        return float(similarity)  # Ensure it's a Python float


if __name__ == "__main__":
    analyzer = StyleAnalyzer()
    print("Style analyzer created. Use this module by importing it.")