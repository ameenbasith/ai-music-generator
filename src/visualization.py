import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json


class MusicVisualizer:
    def __init__(self, output_dir="visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_waveform(self, audio_path, output_path=None):
        """Plot waveform of audio file"""
        y, sr = librosa.load(audio_path, sr=None)

        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title(f'Waveform - {os.path.basename(audio_path)}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close()
            return output_path
        else:
            plt.show()
            plt.close()
            return None

    def plot_spectrogram(self, audio_path, output_path=None):
        """Plot spectrogram of audio file"""
        y, sr = librosa.load(audio_path, sr=None)

        plt.figure(figsize=(12, 6))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram - {os.path.basename(audio_path)}')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close()
            return output_path
        else:
            plt.show()
            plt.close()
            return None

    def plot_chromagram(self, audio_path, output_path=None):
        """Plot chromagram (pitch content) of audio file"""
        y, sr = librosa.load(audio_path, sr=None)

        plt.figure(figsize=(12, 6))
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
        plt.colorbar()
        plt.title(f'Chromagram - {os.path.basename(audio_path)}')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close()
            return output_path
        else:
            plt.show()
            plt.close()
            return None

    def plot_style_comparison(self, features_dir, artist_names, output_path=None):
        """Plot comparison of multiple artist styles using PCA"""
        all_features = []
        artist_labels = []
        feature_names = []

        # Load features for each artist
        for artist in artist_names:
            features_path = os.path.join(features_dir, f"{artist}_features.json")
            if not os.path.exists(features_path):
                print(f"Features not found for {artist}: {features_path}")
                continue

            with open(features_path, 'r') as f:
                features_list = json.load(f)

            for feature_dict in features_list:
                # Create flattened feature vector
                feature_vector = []

                # Add scalar features
                feature_vector.append(feature_dict['spectral_centroid_mean'])
                feature_vector.append(feature_dict['spectral_bandwidth_mean'])
                feature_vector.append(feature_dict['spectral_rolloff_mean'])
                feature_vector.append(feature_dict['tempo'])

                # Add chroma features
                feature_vector.extend(feature_dict['chroma_means'])

                # Add MFCC features
                feature_vector.extend(feature_dict['mfcc_means'])

                all_features.append(feature_vector)
                artist_labels.append(artist)

                # Collect feature names (only need to do once)
                if len(feature_names) == 0:
                    feature_names = ['centroid', 'bandwidth', 'rolloff', 'tempo']
                    feature_names.extend([f'chroma_{i}' for i in range(len(feature_dict['chroma_means']))])
                    feature_names.extend([f'mfcc_{i}' for i in range(len(feature_dict['mfcc_means']))])

        if len(all_features) == 0:
            print("No features found for any artists")
            return None

        # Convert to numpy array
        X = np.array(all_features)

        # Standardize the features
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)

        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_std)

        # Create plot
        plt.figure(figsize=(10, 8))

        # Get unique artists and assign colors
        unique_artists = list(set(artist_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_artists)))

        # Plot each artist's data points
        for i, artist in enumerate(unique_artists):
            mask = [a == artist for a in artist_labels]
            plt.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=[colors[i]],
                label=artist,
                alpha=0.7
            )

        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.title('PCA: Music Style Comparison')
        plt.legend()
        plt.grid(alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close()
            return output_path
        else:
            plt.show()
            plt.close()
            return None

    def visualize_audio_conversion(self, original_path, converted_path, output_dir=None):
        """Create visualizations comparing original and converted audio"""
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'conversion_comparison')

        os.makedirs(output_dir, exist_ok=True)

        # File names for output
        orig_name = os.path.splitext(os.path.basename(original_path))[0]
        conv_name = os.path.splitext(os.path.basename(converted_path))[0]

        # Create waveforms
        orig_wave_path = os.path.join(output_dir, f"{orig_name}_waveform.png")
        conv_wave_path = os.path.join(output_dir, f"{conv_name}_waveform.png")

        self.plot_waveform(original_path, orig_wave_path)
        self.plot_waveform(converted_path, conv_wave_path)

        # Create spectrograms
        orig_spec_path = os.path.join(output_dir, f"{orig_name}_spectrogram.png")
        conv_spec_path = os.path.join(output_dir, f"{conv_name}_spectrogram.png")

        self.plot_spectrogram(original_path, orig_spec_path)
        self.plot_spectrogram(converted_path, conv_spec_path)

        # Create chromagrams
        orig_chroma_path = os.path.join(output_dir, f"{orig_name}_chromagram.png")
        conv_chroma_path = os.path.join(output_dir, f"{conv_name}_chromagram.png")

        self.plot_chromagram(original_path, orig_chroma_path)
        self.plot_chromagram(converted_path, conv_chroma_path)

        # Create a combined figure comparing key features
        self._create_comparison_figure(original_path, converted_path,
                                       os.path.join(output_dir, "comparison.png"))

        print(f"Created visualizations in {output_dir}")
        return output_dir

    def _create_comparison_figure(self, audio1_path, audio2_path, output_path):
        """Create a figure comparing two audio files"""
        # Load audio files
        y1, sr1 = librosa.load(audio1_path, sr=None)
        y2, sr2 = librosa.load(audio2_path, sr=None)

        # Ensure same sample rate
        if sr1 != sr2:
            y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
            sr2 = sr1

        # Create figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 10))

        # Waveforms
        axs[0, 0].set_title('Original Waveform')
        librosa.display.waveshow(y1, sr=sr1, ax=axs[0, 0])

        axs[0, 1].set_title('Converted Waveform')
        librosa.display.waveshow(y2, sr=sr2, ax=axs[0, 1])

        # Spectrograms
        D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
        D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)

        img1 = librosa.display.specshow(D1, sr=sr1, x_axis='time', y_axis='log', ax=axs[1, 0])
        axs[1, 0].set_title('Original Spectrogram')
        fig.colorbar(img1, ax=axs[1, 0], format='%+2.0f dB')

        img2 = librosa.display.specshow(D2, sr=sr2, x_axis='time', y_axis='log', ax=axs[1, 1])
        axs[1, 1].set_title('Converted Spectrogram')
        fig.colorbar(img2, ax=axs[1, 1], format='%+2.0f dB')

        # Chromagrams
        chroma1 = librosa.feature.chroma_cqt(y=y1, sr=sr1)
        chroma2 = librosa.feature.chroma_cqt(y=y2, sr=sr2)

        img3 = librosa.display.specshow(chroma1, sr=sr1, x_axis='time', y_axis='chroma', ax=axs[2, 0])
        axs[2, 0].set_title('Original Chromagram')
        fig.colorbar(img3, ax=axs[2, 0])

        img4 = librosa.display.specshow(chroma2, sr=sr2, x_axis='time', y_axis='chroma', ax=axs[2, 1])
        axs[2, 1].set_title('Converted Chromagram')
        fig.colorbar(img4, ax=axs[2, 1])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        return output_path


if __name__ == "__main__":
    visualizer = MusicVisualizer()
    print("Music visualizer created. Import this module to use it.")