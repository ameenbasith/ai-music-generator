import os
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from demucs.pretrained import get_model
from demucs.audio import AudioFile, save_audio


class AudioProcessor:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.separator = get_model('htdemucs')
        print("Audio processor initialized")

    def load_audio(self, file_path):
        """Load audio file and convert to mono if needed"""
        y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return y, sr

    def separate_vocals_instrumental(self, audio_path, output_dir):
        """Separate vocals from instrumental using Demucs"""
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Load audio using Demucs AudioFile
        audio_file = AudioFile(audio_path, channels=2, samplerate=self.sample_rate)
        wav = audio_file.read()

        # Apply source separation model
        sources = self.separator.separate_tensor(wav)

        # Get the separated sources - Demucs order is [drums, bass, other, vocals]
        vocals = sources[3]  # Vocals

        # Instrumental is everything except vocals (recreate by summing other sources)
        instrumental = sources[0] + sources[1] + sources[2]  # drums + bass + other

        # Base filename without extension
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        # Save separated audio files
        vocal_path = os.path.join(output_dir, f"{base_name}_vocals.wav")
        inst_path = os.path.join(output_dir, f"{base_name}_instrumental.wav")

        save_audio(vocals, vocal_path, self.sample_rate)
        save_audio(instrumental, inst_path, self.sample_rate)

        return vocal_path, inst_path

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

        return {
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'tempo': tempo,
            'chroma_means': chroma_mean.tolist(),
            'mfcc_means': mfcc_means.tolist()
        }


if __name__ == "__main__":
    # Example usage
    processor = AudioProcessor()
    print("Audio processor created. Use this module by importing it.")