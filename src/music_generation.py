import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Config


class MusicStyleDataset(Dataset):
    """Dataset for training music style generation model"""

    def __init__(self, features_dir, artist_name, sequence_length=64):
        self.sequence_length = sequence_length
        self.features = []
        self.artist_name = artist_name

        # Load features
        features_path = os.path.join(features_dir, f"{artist_name}_features.json")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")

        with open(features_path, 'r') as f:
            self.raw_features = json.load(f)

        # Process features into sequences
        self._process_features()

    def _process_features(self):
        """Convert raw features into training sequences"""
        # This is a simplified version - in a real project,
        # you'd create more sophisticated sequences
        for feature_dict in self.raw_features:
            # Create feature vector from dict
            feature_vector = []

            # Add scalar features
            feature_vector.append(feature_dict['spectral_centroid_mean'])
            feature_vector.append(feature_dict['spectral_bandwidth_mean'])
            feature_vector.append(feature_dict['spectral_rolloff_mean'])
            feature_vector.append(feature_dict['tempo'])

            # Add chroma features
            feature_vector.extend(feature_dict['chroma_means'])

            # Add MFCC features
            feature_vector.extend(feature_dict['mfcc_means'])

            # Normalize to range [0,1] for easier training
            feature_vector = np.array(feature_vector)
            feature_vector = (feature_vector - np.min(feature_vector)) / (
                        np.max(feature_vector) - np.min(feature_vector))

            # Add to features list
            self.features.append(feature_vector)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32)


class StyleConditionedMusicGenerator:
    """Model to generate music in the style of a specific artist"""

    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create directory for models
        os.makedirs(model_dir, exist_ok=True)

        # Initialize base model - this is a placeholder
        # In a real implementation, you'd use a more sophisticated model
        # like a transformer trained on music features
        self.base_model = None
        self.artist_models = {}

    def train_artist_model(self, artist_name, features_dir="data/processed_audio",
                           epochs=50, batch_size=16, learning_rate=0.001):
        """Train a model on a specific artist's style"""
        print(f"Training model for {artist_name}...")

        # Create dataset
        dataset = MusicStyleDataset(features_dir, artist_name)
        if len(dataset) == 0:
            print(f"No data available for {artist_name}")
            return False

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create a simple autoencoder model for demonstration
        # In a real project, you'd use a more sophisticated architecture
        input_size = len(dataset[0])

        class SimpleAutoencoder(nn.Module):
            def __init__(self, input_dim):
                super(SimpleAutoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, input_dim),
                    nn.Sigmoid()  # Output between 0 and 1
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

            def encode(self, x):
                return self.encoder(x)

            def decode(self, x):
                return self.decoder(x)

        # Initialize model
        model = SimpleAutoencoder(input_size).to(self.device)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            running_loss = 0.0
            for batch in dataloader:
                batch = batch.to(self.device)

                # Forward pass
                outputs = model(batch)
                loss = criterion(outputs, batch)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Print statistics
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}")

        # Save the model
        model_path = os.path.join(self.model_dir, f"{artist_name}_model.pt")
        torch.save(model.state_dict(), model_path)

        # Store model in memory
        self.artist_models[artist_name] = model

        print(f"Model trained and saved to {model_path}")
        return True

    def load_artist_model(self, artist_name):
        """Load a previously trained artist model"""
        model_path = os.path.join(self.model_dir, f"{artist_name}_model.pt")

        if not os.path.exists(model_path):
            print(f"No trained model found for {artist_name}")
            return False

        # Here we need to know the input size to recreate the model architecture
        # In a real implementation, you would save/load the architecture too
        # This is simplified for the example
        feature_path = os.path.join("data/processed_audio", f"{artist_name}_features.json")
        with open(feature_path, 'r') as f:
            raw_features = json.load(f)

        # Calculate input size
        feature_dict = raw_features[0]
        input_size = (
                4 +  # scalar features
                len(feature_dict['chroma_means']) +
                len(feature_dict['mfcc_means'])
        )

        # Create model instance
        model = SimpleAutoencoder(input_size).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()  # Set to evaluation mode

        # Store model
        self.artist_models[artist_name] = model

        print(f"Loaded model for {artist_name}")
        return True

    def generate_music(self, artist_name, num_samples=1, seed=None):
        """Generate music in the style of the specified artist"""
        # Check if model is loaded
        if artist_name not in self.artist_models:
            success = self.load_artist_model(artist_name)
            if not success:
                print(f"Could not load model for {artist_name}")
                return None

        model = self.artist_models[artist_name]

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Generate random latent vectors
        latent_dim = 32  # Match the latent dimension in our model
        latent_vectors = torch.randn(num_samples, latent_dim).to(self.device)

        # Generate features from latent space
        with torch.no_grad():
            generated_features = model.decode(latent_vectors)

        # Convert to numpy for easier handling
        generated_features = generated_features.cpu().numpy()

        print(f"Generated {num_samples} music feature sets in the style of {artist_name}")
        return generated_features

    def generate_hybrid_style(self, artist1, artist2, blend_ratio=0.5, num_samples=1, seed=None):
        """Generate music that blends styles of two artists"""
        # Load both models
        if artist1 not in self.artist_models:
            self.load_artist_model(artist1)
        if artist2 not in self.artist_models:
            self.load_artist_model(artist2)

        if artist1 not in self.artist_models or artist2 not in self.artist_models:
            print("Could not load one or both artist models")
            return None

        model1 = self.artist_models[artist1]
        model2 = self.artist_models[artist2]

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Generate random latent vectors
        latent_dim = 32
        latent_vectors = torch.randn(num_samples, latent_dim).to(self.device)

        # Generate features from both models
        with torch.no_grad():
            features1 = model1.decode(latent_vectors)
            features2 = model2.decode(latent_vectors)

            # Blend the features
            blended_features = (blend_ratio * features1) + ((1 - blend_ratio) * features2)

        # Convert to numpy
        blended_features = blended_features.cpu().numpy()

        print(
            f"Generated {num_samples} music feature sets blending {artist1} ({blend_ratio:.1f}) and {artist2} ({1 - blend_ratio:.1f})")
        return blended_features


if __name__ == "__main__":
    generator = StyleConditionedMusicGenerator()
    print("Music generator created. Import this module to use it.")