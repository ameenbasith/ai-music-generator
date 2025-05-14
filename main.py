import os
import argparse
import numpy as np
from src.audio_processing import AudioProcessor
from src.style_analysis import StyleAnalyzer
from src.music_generation import StyleConditionedMusicGenerator
from src.voice_conversion import VoiceConverter


def main():
    parser = argparse.ArgumentParser(description="AI Music Generation System")
    # Existing arguments
    parser.add_argument('--analyze', action='store_true', help='Analyze artist styles')
    parser.add_argument('--artist', type=str, help='Artist directory to analyze')
    parser.add_argument('--train', action='store_true', help='Train music generation model')
    parser.add_argument('--generate', action='store_true', help='Generate music')
    parser.add_argument('--blend', type=str, help='Second artist to blend with (requires --artist)')
    parser.add_argument('--ratio', type=float, default=0.5, help='Blend ratio (0.0-1.0)')
    parser.add_argument('--samples', type=int, default=1, help='Number of samples to generate')

    # New voice conversion arguments
    parser.add_argument('--train-voice', action='store_true', help='Train voice conversion model')
    parser.add_argument('--voice-samples', type=str, help='Directory with voice samples')
    parser.add_argument('--voice-name', type=str, help='Name for the voice model')
    parser.add_argument('--convert', action='store_true', help='Convert vocals to target voice')
    parser.add_argument('--vocals', type=str, help='Path to vocals file')
    parser.add_argument('--instrumental', type=str, help='Path to instrumental file')
    parser.add_argument('--target-voice', type=str, help='Target voice model name')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--separate', action='store_true', help='Separate vocals from a song')
    parser.add_argument('--input-song', type=str, help='Input song to separate')

    args = parser.parse_args()

    if args.analyze:
        if not args.artist:
            print("Please specify an artist directory with --artist")
            return

        print(f"Analyzing style for {args.artist}...")
        analyzer = StyleAnalyzer()

        # Path to artist directory
        artist_dir = f"data/reference_artists/{args.artist}"
        if not os.path.exists(artist_dir):
            artist_dir = f"data/your_music"
            if not os.path.exists(artist_dir):
                print(f"Directory not found: {artist_dir}")
                return

        # Extract features and create style vector
        analyzer.extract_features_from_directory(artist_dir, args.artist)
        analyzer.create_style_vector(args.artist)

        print(f"Style analysis complete for {args.artist}")

    elif args.train:
        if not args.artist:
            print("Please specify an artist with --artist")
            return

        print(f"Training music generation model for {args.artist}...")
        generator = StyleConditionedMusicGenerator()
        generator.train_artist_model(args.artist, epochs=30)

    elif args.generate:
        if not args.artist:
            print("Please specify an artist with --artist")
            return

        generator = StyleConditionedMusicGenerator()

        if args.blend:
            print(f"Generating {args.samples} music samples blending {args.artist} and {args.blend}...")
            features = generator.generate_hybrid_style(
                args.artist, args.blend, blend_ratio=args.ratio, num_samples=args.samples
            )
        else:
            print(f"Generating {args.samples} music samples in the style of {args.artist}...")
            features = generator.generate_music(args.artist, num_samples=args.samples)

        if features is not None:
            # For now, just print some stats about the generated features
            print(f"Generated features shape: {features.shape}")
            print(f"Mean: {np.mean(features):.4f}, Min: {np.min(features):.4f}, Max: {np.max(features):.4f}")

            # In a real implementation, these features would be converted to audio
            print("Features generated successfully. In the next step, we'll convert these to audio.")
        # New voice-related functionality
        elif args.train_voice:
            if not args.voice_samples or not args.voice_name:
                print("Please specify --voice-samples DIR and --voice-name NAME")
                return

            print(f"Training voice model {args.voice_name} using samples from {args.voice_samples}")
            converter = VoiceConverter()
            converter.train_voice_model(args.voice_samples, args.voice_name)

        elif args.separate:
            if not args.input_song or not args.output:
                print("Please specify --input-song FILE and --output DIR")
                return

            print(f"Separating vocals and instrumental from {args.input_song}")
            processor = AudioProcessor()
            os.makedirs(args.output, exist_ok=True)
            vocal_path, inst_path = processor.separate_vocals_instrumental(
                args.input_song, args.output
            )
            print(f"Separated to: {vocal_path} and {inst_path}")

        elif args.convert:
            if not args.vocals or not args.target_voice or not args.output:
                print("Please specify --vocals FILE, --target-voice NAME, and --output FILE")
                return

            print(f"Converting vocals to {args.target_voice} style")
            converter = VoiceConverter()

            if args.instrumental:
                print(f"Will mix with instrumental from {args.instrumental}")
                converter.apply_voice_to_music(
                    args.vocals, args.instrumental, args.target_voice, args.output
                )
            else:
                converter.convert_voice(args.vocals, args.target_voice, args.output)

        else:
            print("AI Music Generation System")
            print("\nStyle Analysis and Music Generation:")
            print("  --analyze --artist ARTIST         Analyze artist style")
            print("  --train --artist ARTIST           Train music generation model")
            print("  --generate --artist ARTIST        Generate music in artist style")
            print("  --blend ARTIST2                   Blend two artist styles")

            print("\nVoice Conversion:")
            print("  --train-voice --voice-samples DIR --voice-name NAME   Train voice model")
            print("  --separate --input-song FILE --output DIR             Separate vocals/instrumental")
            print("  --convert --vocals FILE --target-voice NAME --output FILE   Convert voice")
            print("  --instrumental FILE               Add instrumental to voice conversion")

    if __name__ == "__main__":
        main()