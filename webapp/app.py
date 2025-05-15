import os
import sys
import time
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.audio_processing import AudioProcessor
from src.style_analysis import StyleAnalyzer
from src.music_generation import StyleConditionedMusicGenerator
from src.voice_conversion import VoiceConverter
from src.visualization import MusicVisualizer


# Initialize components
@st.cache_resource
def init_components():
    audio_processor = AudioProcessor()
    style_analyzer = StyleAnalyzer()
    music_generator = StyleConditionedMusicGenerator()
    voice_converter = VoiceConverter()
    visualizer = MusicVisualizer()

    return {
        'audio_processor': audio_processor,
        'style_analyzer': style_analyzer,
        'music_generator': music_generator,
        'voice_converter': voice_converter,
        'visualizer': visualizer
    }


# Ensure necessary directories exist
def ensure_dirs():
    os.makedirs("data/raw_audio", exist_ok=True)
    os.makedirs("data/processed_audio", exist_ok=True)
    os.makedirs("data/your_music", exist_ok=True)
    os.makedirs("data/reference_artists", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/voice", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("webapp/uploads", exist_ok=True)
    os.makedirs("webapp/output", exist_ok=True)


# Save uploaded file
def save_uploaded_file(uploaded_file, directory="webapp/uploads"):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


# Main app
def main():
    ensure_dirs()
    components = init_components()

    st.title("AI Music Generation System")
    st.write("Generate music in your style or other artists' styles, and transform vocals.")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a mode",
        ["Home", "Style Analysis", "Music Generation", "Voice Conversion", "Audio Visualization"]
    )

    # Home page
    if app_mode == "Home":
        st.header("Welcome to the AI Music Generation System")
        st.write("""
        This system allows you to:

        1. **Analyze Music Styles** - Extract style features from your music or reference artists
        2. **Generate Original Music** - Create new music in your style or blend with other styles
        3. **Convert Vocal Styles** - Transform vocals to match your voice or other artists
        4. **Visualize Audio** - Create visual representations of audio characteristics

        Get started by selecting a mode from the sidebar.
        """)

        st.subheader("How it Works")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Style Analysis**")
            st.write("Extracts musical features and creates style profiles")

        with col2:
            st.write("**Music Generation**")
            st.write("Creates original compositions in specific styles")

        with col3:
            st.write("**Voice Conversion**")
            st.write("Transforms vocals to match different voice characteristics")

    # Style Analysis page
    elif app_mode == "Style Analysis":
        st.header("Music Style Analysis")
        st.write("Upload music to analyze and extract style characteristics.")

        analysis_mode = st.radio(
            "Select Analysis Mode",
            ["Analyze Single Track", "Analyze Artist Directory", "Compare Styles"]
        )

        if analysis_mode == "Analyze Single Track":
            uploaded_file = st.file_uploader("Upload a music file (WAV or MP3)", type=["wav", "mp3"])

            if uploaded_file:
                st.audio(uploaded_file)

                if st.button("Analyze Track"):
                    with st.spinner("Analyzing audio..."):
                        # Save uploaded file
                        file_path = save_uploaded_file(uploaded_file)

                        # Load audio
                        audio, sr = components['audio_processor'].load_audio(file_path)

                        # Extract features
                        features = components['audio_processor'].extract_features(audio, sr)

                        # Display features
                        st.subheader("Audio Features")
                        st.json(features)

                        # Create visualizations
                        st.subheader("Audio Visualizations")

                        # Create waveform
                        plt.figure(figsize=(10, 4))
                        librosa.display.waveshow(audio, sr=sr)
                        plt.title('Waveform')
                        st.pyplot(plt)

                        # Create spectrogram
                        plt.figure(figsize=(10, 4))
                        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
                        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
                        plt.colorbar(format='%+2.0f dB')
                        plt.title('Spectrogram')
                        st.pyplot(plt)

                        # Create chromagram
                        plt.figure(figsize=(10, 4))
                        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
                        librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
                        plt.colorbar()
                        plt.title('Chromagram')
                        st.pyplot(plt)

        elif analysis_mode == "Analyze Artist Directory":
            st.write("This mode analyzes a directory of music files to create an artist style profile.")
            st.write("For demonstration purposes, use the single track uploader multiple times.")

            artist_name = st.text_input("Artist Name")
            uploaded_files = st.file_uploader("Upload music files (WAV or MP3)",
                                              type=["wav", "mp3"], accept_multiple_files=True)

            if uploaded_files and artist_name:
                # Create artist directory
                artist_dir = os.path.join("data/reference_artists", artist_name)
                os.makedirs(artist_dir, exist_ok=True)

                # Save all uploaded files
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(artist_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)

                st.write(f"Uploaded {len(file_paths)} files to {artist_dir}")

                if st.button("Create Artist Style Profile"):
                    with st.spinner(f"Analyzing {artist_name}'s style..."):
                        # Extract features and create style vector
                        components['style_analyzer'].extract_features_from_directory(artist_dir, artist_name)
                        style_vector = components['style_analyzer'].create_style_vector(artist_name)

                        st.success(f"Created style profile for {artist_name}")

        elif analysis_mode == "Compare Styles":
            st.write("Compare musical styles between artists.")

            # Get available artist profiles
            processed_dir = "data/processed_audio"
            available_artists = []

            if os.path.exists(processed_dir):
                for file in os.listdir(processed_dir):
                    if file.endswith("_style_vector.json"):
                        artist_name = file.replace("_style_vector.json", "")
                        available_artists.append(artist_name)

            if len(available_artists) < 2:
                st.warning("You need at least two artist profiles for comparison. Please analyze artists first.")
            else:
                selected_artists = st.multiselect("Select artists to compare", available_artists)

                if len(selected_artists) >= 2 and st.button("Compare Styles"):
                    with st.spinner("Generating style comparison..."):
                        # Create visualization
                        output_path = os.path.join("visualizations", "style_comparison.png")
                        components['visualizer'].plot_style_comparison(
                            "data/processed_audio", selected_artists, output_path
                        )

                        # Display image
                        st.image(output_path, caption="Style Comparison (PCA)")

                        # Calculate similarity scores
                        st.subheader("Style Similarity Scores")

                        for i in range(len(selected_artists)):
                            for j in range(i + 1, len(selected_artists)):
                                artist1 = selected_artists[i]
                                artist2 = selected_artists[j]

                                similarity = components['style_analyzer'].compare_styles(artist1, artist2)
                                st.write(f"{artist1} vs {artist2}: {similarity:.2f}")

    # Music Generation page
    elif app_mode == "Music Generation":
        st.header("AI Music Generation")
        st.write("Generate original music in different styles.")

        # Get available artist profiles
        processed_dir = "data/processed_audio"
        available_artists = []

        if os.path.exists(processed_dir):
            for file in os.listdir(processed_dir):
                if file.endswith("_style_vector.json"):
                    artist_name = file.replace("_style_vector.json", "")
                    available_artists.append(artist_name)

        if len(available_artists) == 0:
            st.warning("No artist profiles available. Please analyze artists first in the Style Analysis section.")
        else:
            generation_mode = st.radio(
                "Select Generation Mode",
                ["Single Artist Style", "Blend Multiple Styles"]
            )

            if generation_mode == "Single Artist Style":
                selected_artist = st.selectbox("Select Artist Style", available_artists)
                num_samples = st.slider("Number of Samples to Generate", 1, 5, 1)

                if st.button("Generate Music"):
                    with st.spinner(f"Generating music in the style of {selected_artist}..."):
                        # Train model if not already trained
                        if not os.path.exists(os.path.join("models", f"{selected_artist}_model.pt")):
                            st.info(f"Training model for {selected_artist}...")
                            components['music_generator'].train_artist_model(selected_artist)

                        # Generate features
                        features = components['music_generator'].generate_music(
                            selected_artist, num_samples=num_samples
                        )

                        if features is not None:
                            st.success(f"Generated {num_samples} music feature sets")

                            # Display feature statistics
                            st.subheader("Generated Feature Statistics")
                            for i, feature_set in enumerate(features):
                                st.write(f"Sample {i + 1}:")
                                st.write(f"Mean: {np.mean(feature_set):.4f}")
                                st.write(f"Min: {np.min(feature_set):.4f}")
                                st.write(f"Max: {np.max(feature_set):.4f}")

                            st.info(
                                "Note: In a complete implementation, these features would be converted to actual audio.")
                        else:
                            st.error("Failed to generate music features")

            elif generation_mode == "Blend Multiple Styles":
                if len(available_artists) < 2:
                    st.warning("You need at least two artist profiles for blending. Please analyze more artists.")
                else:
                    artist1 = st.selectbox("First Artist Style", available_artists)
                    artist2 = st.selectbox("Second Artist Style",
                                           [a for a in available_artists if a != artist1])

                    blend_ratio = st.slider("Blend Ratio", 0.0, 1.0, 0.5,
                                            help="0.0 = 100% Second Artist, 1.0 = 100% First Artist")

                    num_samples = st.slider("Number of Samples to Generate", 1, 5, 1)

                    if st.button("Generate Blended Music"):
                        with st.spinner(f"Generating music blending {artist1} and {artist2}..."):
                            # Train models if not already trained
                            for artist in [artist1, artist2]:
                                if not os.path.exists(os.path.join("models", f"{artist}_model.pt")):
                                    st.info(f"Training model for {artist}...")
                                    components['music_generator'].train_artist_model(artist)

                            # Generate blended features
                            features = components['music_generator'].generate_hybrid_style(
                                artist1, artist2, blend_ratio=blend_ratio, num_samples=num_samples
                            )

                            if features is not None:
                                st.success(f"Generated {num_samples} blended music feature sets")

                                # Display feature statistics
                                st.subheader("Generated Feature Statistics")
                                for i, feature_set in enumerate(features):
                                    st.write(f"Sample {i + 1}:")
                                    st.write(f"Mean: {np.mean(feature_set):.4f}")
                                    st.write(f"Min: {np.min(feature_set):.4f}")
                                    st.write(f"Max: {np.max(feature_set):.4f}")

                                st.info(
                                    "Note: In a complete implementation, these features would be converted to actual audio.")
                            else:
                                st.error("Failed to generate blended music features")

    # Voice Conversion page
    elif app_mode == "Voice Conversion":
        st.header("Voice Conversion")
        st.write("Train voice models and convert vocals to match different voices.")

        conversion_mode = st.radio(
            "Select Mode",
            ["Train Voice Model", "Separate Vocals", "Convert Voice"]
        )

        if conversion_mode == "Train Voice Model":
            st.subheader("Train a Voice Model")
            voice_name = st.text_input("Voice Model Name")
            uploaded_files = st.file_uploader("Upload voice samples (WAV or MP3)",
                                              type=["wav", "mp3"], accept_multiple_files=True)

            if uploaded_files and voice_name:
                # Create voice samples directory
                voice_dir = os.path.join("data/voice_samples", voice_name)
                os.makedirs(voice_dir, exist_ok=True)

                # Save all uploaded files
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(voice_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)

                st.write(f"Uploaded {len(file_paths)} voice samples to {voice_dir}")

                if st.button("Train Voice Model"):
                    with st.spinner(f"Training voice model for {voice_name}..."):
                        success = components['voice_converter'].train_voice_model(voice_dir, voice_name)

                        if success:
                            st.success(f"Trained voice model for {voice_name}")
                        else:
                            st.error("Failed to train voice model")

        elif conversion_mode == "Separate Vocals":
            st.subheader("Separate Vocals from Instrumental")
            uploaded_file = st.file_uploader("Upload a song (WAV or MP3)", type=["wav", "mp3"])

            if uploaded_file:
                st.audio(uploaded_file)

                if st.button("Separate Vocals"):
                    with st.spinner("Separating vocals from instrumental..."):
                        # Save uploaded file
                        file_path = save_uploaded_file(uploaded_file)

                        # Separate vocals
                        output_dir = os.path.join("webapp/output", "separated")
                        os.makedirs(output_dir, exist_ok=True)

                        vocal_path, inst_path = components['audio_processor'].separate_vocals_instrumental(
                            file_path, output_dir
                        )

                        # Display results
                        st.success("Separation complete")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Vocals")
                            st.audio(vocal_path)

                        with col2:
                            st.subheader("Instrumental")
                            st.audio(inst_path)

        elif conversion_mode == "Convert Voice":
            st.subheader("Convert Voice to Target Style")

            # Get available voice models
            voice_models = []
            voice_dir = "models/voice"

            if os.path.exists(voice_dir):
                for file in os.listdir(voice_dir):
                    if file.endswith("_profile.npy"):
                        model_name = file.replace("_profile.npy", "")
                        voice_models.append(model_name)

            # Upload vocal file
            uploaded_vocals = st.file_uploader("Upload vocals (WAV)", type=["wav"])

            if len(voice_models) == 0:
                st.warning("No voice models available. Please train a voice model first.")
            elif uploaded_vocals:
                st.audio(uploaded_vocals)

                # Select target voice model
                target_voice = st.selectbox("Select Target Voice", voice_models)

                # Optional instrumental upload
                st.write("Optional: Upload instrumental to mix with converted vocals")
                uploaded_instrumental = st.file_uploader("Upload instrumental (WAV)", type=["wav"])

                if uploaded_instrumental:
                    st.audio(uploaded_instrumental)

                if st.button("Convert Voice"):
                    with st.spinner("Converting voice..."):
                        # Save uploaded files
                        vocals_path = save_uploaded_file(uploaded_vocals)

                        # Convert voice
                        output_dir = os.path.join("webapp/output", "converted")
                        os.makedirs(output_dir, exist_ok=True)

                        if uploaded_instrumental:
                            instrumental_path = save_uploaded_file(uploaded_instrumental)
                            output_path = os.path.join(output_dir, f"converted_{target_voice}_mix.wav")

                            success = components['voice_converter'].apply_voice_to_music(
                                vocals_path, instrumental_path, target_voice, output_path
                            )
                        else:
                            output_path = os.path.join(output_dir, f"converted_{target_voice}.wav")
                            success = components['voice_converter'].convert_voice(
                                vocals_path, target_voice, output_path
                            )

                        if success:
                            st.success("Voice conversion complete")
                            st.audio(output_path)

                            # Create visualizations
                            vis_output_dir = components['visualizer'].visualize_audio_conversion(
                                vocals_path, output_path
                            )

                            st.subheader("Before/After Comparison")
                            st.image(os.path.join(vis_output_dir, "comparison.png"))
                        else:
                            st.error("Voice conversion failed")

    # Audio Visualization page
    elif app_mode == "Audio Visualization":
        st.header("Audio Visualization")
        st.write("Create visual representations of audio characteristics.")

        visualization_mode = st.radio(
            "Select Visualization Mode",
            ["Single Audio File", "Compare Two Audio Files"]
        )

        if visualization_mode == "Single Audio File":
            uploaded_file = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"])

            if uploaded_file:
                st.audio(uploaded_file)

                visualization_type = st.multiselect(
                    "Select Visualization Types",
                    ["Waveform", "Spectrogram", "Chromagram"],
                    ["Waveform", "Spectrogram", "Chromagram"]
                )

                if st.button("Generate Visualizations"):
                    if not visualization_type:
                        st.warning("Please select at least one visualization type")
                    else:
                        with st.spinner("Generating visualizations..."):
                            # Save uploaded file
                            file_path = save_uploaded_file(uploaded_file)

                            # Generate visualizations
                            for vis_type in visualization_type:
                                if vis_type == "Waveform":
                                    st.subheader("Waveform")
                                    fig = plt.figure(figsize=(10, 4))
                                    audio, sr = librosa.load(file_path, sr=None)
                                    librosa.display.waveshow(audio, sr=sr)
                                    plt.title("Waveform")
                                    st.pyplot(fig)

                                elif vis_type == "Spectrogram":
                                    st.subheader("Spectrogram")
                                    fig = plt.figure(figsize=(10, 4))
                                    audio, sr = librosa.load(file_path, sr=None)
                                    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
                                    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
                                    plt.colorbar(format='%+2.0f dB')
                                    plt.title("Spectrogram")
                                    st.pyplot(fig)

                                elif vis_type == "Chromagram":
                                    st.subheader("Chromagram")
                                    fig = plt.figure(figsize=(10, 4))
                                    audio, sr = librosa.load(file_path, sr=None)
                                    chroma = librosa

                                elif vis_type == "Chromagram":
                                    st.subheader("Chromagram")
                                    fig = plt.figure(figsize=(10, 4))
                                    audio, sr = librosa.load(file_path, sr=None)
                                    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
                                    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
                                    plt.colorbar()
                                    plt.title("Chromagram")
                                    st.pyplot(fig)
                elif visualization_mode == "Compare Two Audio Files":
                    st.subheader("Compare Two Audio Files")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("First Audio File")
                        uploaded_file1 = st.file_uploader("Upload first audio file", type=["wav", "mp3"])

                        if uploaded_file1:
                            st.audio(uploaded_file1)

                    with col2:
                        st.write("Second Audio File")
                        uploaded_file2 = st.file_uploader("Upload second audio file", type=["wav", "mp3"])

                        if uploaded_file2:
                            st.audio(uploaded_file2)

                    if uploaded_file1 and uploaded_file2 and st.button("Compare Audio Files"):
                        with st.spinner("Generating comparison..."):
                            # Save uploaded files
                            file_path1 = save_uploaded_file(uploaded_file1)
                            file_path2 = save_uploaded_file(uploaded_file2)

                            # Create output directory
                            output_dir = os.path.join("visualizations", "comparison")
                            os.makedirs(output_dir, exist_ok=True)

                            # Generate comparison
                            comparison_path = components['visualizer']._create_comparison_figure(
                                file_path1, file_path2, os.path.join(output_dir, "comparison.png")
                            )

                            # Display comparison
                            st.image(comparison_path, caption="Audio Comparison")

if __name__ == "__main__":
    main()