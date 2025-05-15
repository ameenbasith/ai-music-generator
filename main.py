import os
import argparse
import numpy as np
from src.audio_processing import AudioProcessor
from src.style_analysis import StyleAnalyzer
from src.music_generation import StyleConditionedMusicGenerator
from src.voice_conversion import VoiceConverter
from src.visualization import MusicVisualizer


def main():
    parser = argparse.ArgumentParser(description="AI Music Generation System")

    # Style analysis arguments
    parser.add_argument('--analyze', action='store_true', help='Analyze artist styles')
    parser.add_argument('--artist', type=str, help='Artist directory to analyze')

    # Music generation arguments
    parser.add_argument('--train', action='store_true', help='Train music generation model')
    parser.add_argument('--generate', action='store_true', help='Generate music')
    parser.add_argument('--blend', type=str, help='Second artist to blend with (requires --artist)')
    parser.add_argument('--ratio', type=float, default=0.5, help='Blend ratio (0.0-1.0)')
    parser.add_argument('--samples', type=int, default=1, help='Number of samples to generate')

    # Voice conversion arguments
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

    # Visualization arguments
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--compare-styles', action='store_true', help='Compare artist styles visually')
    parser.add_argument('--artists', type=str, nargs='+', help='List of artists to compare')
    parser.add_argument('--compare-audio', action='store_true', help='Compare two audio files')
    parser.add_argument('--audio1', type=str, help='First audio file to compare')
    parser.add_argument('--audio2', type=str, help='Second audio file to compare')
    parser.add_argument('--visualize-type', type=str, choices=['waveform', 'spectrogram', 'chromagram', 'all'],
                        default='all', help='Type of visualization to create')

    args = parser.parse_args()

    # Style Analysis
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

    # Music Generation
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

    # Voice Conversion
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

    # Visualization
    elif args.visualize:
        visualizer = MusicVisualizer()

        if args.compare_styles:
            if not args.artists or len(args.artists) < 2:
                print("Please specify at least two artists to compare with --artists")
                return

            print(f"Creating style comparison visualization for artists: {', '.join(args.artists)}")
            output_path = os.path.join("visualizations", "style_comparison.png")
            visualizer.plot_style_comparison("data/processed_audio", args.artists, output_path)
            print(f"Style comparison saved to {output_path}")

        elif args.compare_audio:
            if not args.audio1 or not args.audio2:
                print("Please specify two audio files to compare with --audio1 and --audio2")
                return

            print(f"Comparing audio files: {args.audio1} and {args.audio2}")
            output_dir = os.path.join("visualizations", "audio_comparison")
            os.makedirs(output_dir, exist_ok=True)

            if args.visualize_type == 'waveform' or args.visualize_type == 'all':
                visualizer.plot_waveform(args.audio1, os.path.join(output_dir, "audio1_waveform.png"))
                visualizer.plot_waveform(args.audio2, os.path.join(output_dir, "audio2_waveform.png"))

            if args.visualize_type == 'spectrogram' or args.visualize_type == 'all':
                visualizer.plot_spectrogram(args.audio1, os.path.join(output_dir, "audio1_spectrogram.png"))
                visualizer.plot_spectrogram(args.audio2, os.path.join(output_dir, "audio2_spectrogram.png"))

            if args.visualize_type == 'chromagram' or args.visualize_type == 'all':
                visualizer.plot_chromagram(args.audio1, os.path.join(output_dir, "audio1_chromagram.png"))
                visualizer.plot_chromagram(args.audio2, os.path.join(output_dir, "audio2_chromagram.png"))

            # Create comparison figure
            visualizer._create_comparison_figure(args.audio1, args.audio2,
                                                 os.path.join(output_dir, "comparison.png"))

            print(f"Audio comparison visualizations saved to {output_dir}")

        else:
            if not args.audio1:
                print("Please specify an audio file to visualize with --audio1")
                return

            print(f"Creating visualizations for audio: {args.audio1}")
            output_dir = os.path.join("visualizations", "single_audio")
            os.makedirs(output_dir, exist_ok=True)

            file_name = os.path.splitext(os.path.basename(args.audio1))[0]

            if args.visualize_type == 'waveform' or args.visualize_type == 'all':
                path = visualizer.plot_waveform(args.audio1,
                                                os.path.join(output_dir, f"{file_name}_waveform.png"))
                print(f"Waveform saved to {path}")

            if args.visualize_type == 'spectrogram' or args.visualize_type == 'all':
                path = visualizer.plot_spectrogram(args.audio1,
                                                   os.path.join(output_dir, f"{file_name}_spectrogram.png"))
                print(f"Spectrogram saved to {path}")

            if args.visualize_type == 'chromagram' or args.visualize_type == 'all':
                path = visualizer.plot_chromagram(args.audio1,
                                                  os.path.join(output_dir, f"{file_name}_chromagram.png"))
                print(f"Chromagram saved to {path}")

    # Show help if no arguments provided
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

        print("\nVisualizations:")
        print("  --visualize --audio1 FILE         Visualize audio file")
        print("  --visualize-type TYPE             Type of visualization (waveform, spectrogram, chromagram, all)")
        print("  --compare-audio --audio1 FILE1 --audio2 FILE2         Compare two audio files")
        print("  --compare-styles --artists ARTIST1 ARTIST2 [...]      Compare artist styles")


if __name__ == "__main__":
    main()