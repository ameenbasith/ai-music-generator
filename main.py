import os
import argparse
from src.audio_processing import AudioProcessor
from src.style_analysis import StyleAnalyzer


def main():
    parser = argparse.ArgumentParser(description="AI Music Generation System")
    parser.add_argument('--analyze', action='store_true', help='Analyze artist styles')
    parser.add_argument('--artist', type=str, help='Artist directory to analyze')

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
    else:
        print("AI Music Generation System")
        print("Use --analyze to analyze artist styles")


if __name__ == "__main__":
    main()