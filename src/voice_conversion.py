import os
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class VoiceConverter:
    def __init__(self, model_dir="models/voice"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load voice encoder model (for voice identification)
        # In a real implementation, you would use a more sophisticated model
        # trained specifically for voice conversion
        try:
            # Load a pre-trained speech recognition model as a feature extractor
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
            print("Loaded voice encoder model")
        except Exception as e:
            print(f"Error loading voice encoder model: {e}")
            self.processor = None
            self.model = None

        # In a production version, you would include:
        # 1. A voice encoder (to extract voice characteristics)
        # 2. A voice decoder (to generate new voice with desired characteristics)
        # 3. A pitch shifter for adjusting pitch
        # 4. A timbre transfer model

    def extract_voice_features(self, audio_path):
        """Extract voice features from audio file"""
        try:
            # Load audio
            speech, sr = librosa.load(audio_path, sr=16000)

            # Ensure model is loaded
            if self.processor is None or self.model is None:
                print("Voice encoder model not loaded")
                return None

            # Process audio
            inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract features (without computing CTC loss)
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get hidden states as voice features
            voice_features = outputs.logits.mean(dim=1).cpu().numpy()

            print(f"Extracted voice features from {audio_path}")
            return voice_features

        except Exception as e:
            print(f"Error extracting voice features: {e}")
            return None

    def train_voice_model(self, voice_samples_dir, output_model_name):
        """Train a voice conversion model using samples from a directory"""
        # Get all audio files
        audio_files = []
        for root, _, files in os.walk(voice_samples_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3')):
                    audio_files.append(os.path.join(root, file))

        if len(audio_files) == 0:
            print(f"No audio files found in {voice_samples_dir}")
            return False

        print(f"Found {len(audio_files)} voice samples")

        # Extract features from each file
        voice_features = []
        for audio_file in audio_files:
            features = self.extract_voice_features(audio_file)
            if features is not None:
                voice_features.append(features)

        if len(voice_features) == 0:
            print("Failed to extract any voice features")
            return False

        # Create average voice profile
        voice_profile = np.mean(voice_features, axis=0)

        # Save voice profile
        profile_path = os.path.join(self.model_dir, f"{output_model_name}_profile.npy")
        np.save(profile_path, voice_profile)

        print(f"Trained voice model and saved to {profile_path}")

        # In a real implementation, you would train a more sophisticated model here
        # For instance, using a voice conversion autoencoder or GAN

        return True

    def convert_voice(self, input_audio_path, target_voice_model, output_path):
        """Convert vocals to target voice"""
        # Load target voice profile
        profile_path = os.path.join(self.model_dir, f"{target_voice_model}_profile.npy")
        if not os.path.exists(profile_path):
            print(f"Voice profile not found: {profile_path}")
            return False

        voice_profile = np.load(profile_path)

        # Load input audio
        audio, sr = librosa.load(input_audio_path, sr=16000)

        # In a real implementation, you would:
        # 1. Separate vocals from music if needed
        # 2. Extract phonemes/linguistic content
        # 3. Apply voice conversion model
        # 4. Resynthesize with target voice characteristics

        # For this demonstration, we'll simply apply basic audio effects
        # to simulate voice conversion (this is NOT actual voice conversion)

        # Simple simulation of voice conversion (pitch shift + formant shift)
        # In a real project, you'd use a neural model here
        print(f"Converting voice in {input_audio_path} to {target_voice_model} style")
        print("This is a simplified simulation - real voice conversion requires a trained neural model")

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Placeholder for now - just copy the file
        # In a real implementation, this would actually transform the voice
        sf.write(output_path, audio, sr)

        print(f"Simulated voice conversion complete: {output_path}")
        return True

    def apply_voice_to_music(self, vocals_path, instrumental_path, target_voice_model, output_path):
        """Apply target voice model to vocals and mix with instrumental"""
        # First convert the vocals
        converted_vocals_path = os.path.join(os.path.dirname(output_path), "converted_vocals.wav")
        success = self.convert_voice(vocals_path, target_voice_model, converted_vocals_path)

        if not success:
            print("Voice conversion failed")
            return False

        # Load converted vocals and instrumental
        vocals, sr_v = librosa.load(converted_vocals_path, sr=None)
        instrumental, sr_i = librosa.load(instrumental_path, sr=None)

        # Ensure same sample rate
        if sr_v != sr_i:
            print(f"Resampling instrumental from {sr_i} to {sr_v}")
            instrumental = librosa.resample(instrumental, orig_sr=sr_i, target_sr=sr_v)
            sr_i = sr_v

        # Ensure same length
        if len(vocals) > len(instrumental):
            vocals = vocals[:len(instrumental)]
        elif len(instrumental) > len(vocals):
            # Pad vocals with silence
            vocals = np.pad(vocals, (0, len(instrumental) - len(vocals)))

        # Mix vocals and instrumental
        # Vocal volume can be adjusted here
        vocal_volume = 1.0
        instrumental_volume = 0.8

        mix = (vocals * vocal_volume) + (instrumental * instrumental_volume)

        # Normalize to prevent clipping
        max_val = np.max(np.abs(mix))
        if max_val > 1.0:
            mix = mix / max_val * 0.95

        # Save mixed audio
        sf.write(output_path, mix, sr_v)

        print(f"Mixed converted vocals with instrumental: {output_path}")
        return True


if __name__ == "__main__":
    converter = VoiceConverter()
    print("Voice converter created. Import this module to use it.")