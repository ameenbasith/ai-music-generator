import os
import subprocess
import webbrowser
import time
import sys


def main():
    """Launch the web application"""
    print("Starting AI Music Generation System Web Interface...")

    # Check if streamlit is installed
    try:
        import streamlit
        print("Streamlit is installed. Launching web interface...")
    except ImportError:
        print("Streamlit is not installed. Installing streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])

    # Launch the streamlit app
    webapp_path = os.path.join("webapp", "app.py")

    # Make sure the webapp directory exists
    os.makedirs("webapp", exist_ok=True)

    # Check if app.py exists
    if not os.path.exists(webapp_path):
        print(f"Error: {webapp_path} not found. Make sure you've created the webapp directory and app.py file.")
        return

    # Start streamlit
    print("Launching web interface. This may take a moment...")

    # Use subprocess to start streamlit in a new process
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", webapp_path, "--server.port=8501"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait a moment for the server to start
    time.sleep(3)

    # Open the web browser
    url = "http://localhost:8501"
    print(f"Opening web interface at {url}")
    webbrowser.open(url)

    print("\nWeb interface is running. Close this terminal window to shut down the server.")

    # Keep the script running to maintain the server
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down web interface...")
        process.terminate()
        process.wait()
        print("Web interface shut down.")


if __name__ == "__main__":
    main()