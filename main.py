import os
import argparse
import numpy as np
from src.audio_processing import AudioProcessor
from src.style_analysis import StyleAnalyzer
from src.music_generation import StyleConditionedMusicGenerator


def main():
    parser = argparse.ArgumentParser(description="AI Music Generation System")
    parser.add_argument('--analyze', action='store_true', help='Analyze artist styles')
    parser.add_argument('--artist', type=str, help='Artist directory to analyze')
    parser.add_argument('--train', action='store_true', help='Train music generation model')
    parser.add_argument('--generate', action='store_true', help='Generate music')
    parser.add_argument('--blend', type=str, help='Second artist to blend with (requires --artist)')
    parser.add_argument('--ratio', type=float, default=0.5, help='Blend ratio (0.0-1.0)')
    parser.add_argument('--samples', type=int, default=1, help='Number of samples to generate')

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
    else:
        print("AI Music Generation System")
        print("Use --analyze to analyze artist styles")
        print("Use --train --artist ARTIST to train a generation model")
        print("Use --generate --artist ARTIST to generate music")
        print("Add --blend ARTIST2 to blend two artists' styles")


if __name__ == "__main__":
    main()