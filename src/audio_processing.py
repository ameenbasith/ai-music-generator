import os
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import torch
import torchaudio
import subprocess
import warnings
import tempfile
import shutil


class AudioProcessor:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        print("Audio processor initialized")

    def load_audio(self, file_path):
        """Load audio file and convert to mono if needed"""
        y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return y, sr

    def separate_vocals_instrumental(self, audio_path, output_dir):
        """Separate vocals from instrumental using Spleeter instead of Demucs"""
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Create a unique temporary directory for spleeter output
            with tempfile.TemporaryDirectory() as temp_dir:
                # File paths
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                vocal_path = os.path.join(output_dir, f"{base_name}_vocals.wav")
                inst_path = os.path.join(output_dir, f"{base_name}_instrumental.wav")

                # Run spleeter as a subprocess (make sure spleeter is installed)
                print(f"Separating vocals using spleeter (this may take a moment)...")
                try:
                    # Try using spleeter if installed
                    subprocess.run([
                        "spleeter", "separate",
                        "-p", "spleeter:2stems",
                        "-o", temp_dir,
                        audio_path
                    ], check=True)

                    # Copy the separated files to final destination
                    spleeter_vocal = os.path.join(temp_dir, base_name, "vocals.wav")
                    spleeter_inst = os.path.join(temp_dir, base_name, "accompaniment.wav")

                    shutil.copy(spleeter_vocal, vocal_path)
                    shutil.copy(spleeter_inst, inst_path)

                    print(f"Saved separated vocals to {vocal_path}")
                    print(f"Saved instrumental to {inst_path}")

                except (subprocess.SubprocessError, FileNotFoundError):
                    # If spleeter fails or isn't installed, fall back to a simple frequency-based separation
                    print("Spleeter not available or failed. Using fallback separation method...")
                    self._fallback_separation(audio_path, vocal_path, inst_path)

            return vocal_path, inst_path

        except Exception as e:
            print(f"Error during source separation: {e}")
            # If all else fails, just copy the original file to both outputs
            sf.write(vocal_path, np.zeros(1000), self.sample_rate)  # Empty file to avoid errors
            shutil.copy(audio_path, inst_path)  # Just use original for instrumental
            return vocal_path, inst_path

    def _fallback_separation(self, audio_path, vocal_path, inst_path):
        """Fallback method for audio separation when Spleeter isn't available.
        This is a very simple approach that assumes vocals are dominant in mid-range frequencies."""
        try:
            # Load the audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Compute the Short-Time Fourier Transform (STFT)
            D = librosa.stft(y)

            # Get magnitude and phase
            magnitude, phase = librosa.magphase(D)

            # Create a frequency mask for vocals (very simple approach)
            # Most vocals are in the mid-frequency range
            freq_bins = magnitude.shape[0]
            vocal_mask = np.zeros_like(magnitude)

            # Focus on mid frequencies for vocals (around 200Hz-3kHz)
            low_bin = int(200 * freq_bins / (sr / 2))
            high_bin = int(3000 * freq_bins / (sr / 2))
            vocal_mask[low_bin:high_bin, :] = 1

            # Apply masks to extract vocals and instrumentals
            vocals_mag = magnitude * vocal_mask
            inst_mag = magnitude * (1 - vocal_mask * 0.8)  # Allow some overlap

            # Reconstruct audio with phase information
            vocals = librosa.istft(vocals_mag * phase)
            instrumental = librosa.istft(inst_mag * phase)

            # Normalize
            vocals = vocals / np.max(np.abs(vocals))
            instrumental = instrumental / np.max(np.abs(instrumental))

            # Save files
            sf.write(vocal_path, vocals, sr)
            sf.write(inst_path, instrumental, sr)

            print(f"Used fallback separation method - quality will be limited")
            print(f"Saved separated vocals to {vocal_path}")
            print(f"Saved instrumental to {inst_path}")

        except Exception as e:
            print(f"Fallback separation failed: {e}")
            # If even this fails, create placeholder files
            sf.write(vocal_path, np.zeros(1000), self.sample_rate)
            sf.write(inst_path, np.zeros(1000), self.sample_rate)

    def extract_features(self, audio, sr):
        """Extract audio features for style analysis"""
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]

        # Rhythmic features
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

        # Tonal features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)

        # Convert NumPy arrays to Python lists for JSON serialization
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'tempo': float(tempo),
            'chroma_means': chroma_mean.tolist(),  # Convert to list
            'mfcc_means': mfcc_means.tolist()  # Convert to list
        }


if __name__ == "__main__":
    # Example usage
    processor = AudioProcessor()
    print("Audio processor created. Use this module by importing it.")