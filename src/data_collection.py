import os
import requests
import youtube_dl
import argparse
from pathlib import Path


def create_directories():
    """Create necessary directories for data storage"""
    directories = [
        "data/raw_audio",
        "data/processed_audio",
        "data/your_music",
        "data/reference_artists/kanye",
        "data/reference_artists/weeknd",
        "data/reference_artists/frank_ocean"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def download_audio_from_youtube(url, output_path, artist_name=None):
    """Download audio from YouTube URL"""
    if artist_name:
        output_path = os.path.join(output_path, artist_name)

    os.makedirs(output_path, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print(f"Downloaded audio to {output_path}")


if __name__ == "__main__":
    create_directories()

    parser = argparse.ArgumentParser(description="Download music for AI training")
    parser.add_argument("--url", type=str, help="YouTube URL to download")
    parser.add_argument("--artist", type=str, help="Artist name for categorization")

    args = parser.parse_args()

    if args.url:
        download_audio_from_youtube(args.url, "data/raw_audio", args.artist)
    else:
        print("No URL provided. Created directories only.")