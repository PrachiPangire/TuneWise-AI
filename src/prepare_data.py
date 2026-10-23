
import pandas as pd

# Load the Spotify dataset
df = pd.read_csv("data/dataset.csv")

# Audio features used to predict music genre
FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

TARGET = "track_genre"

# Keep only the features and target needed for classification
data = df[FEATURES + [TARGET]].copy()

# Remove rows with missing values in selected columns
data = data.dropna(subset=FEATURES + [TARGET])

print("Data preparation successful!")
print("Dataset shape:", data.shape)
print("Number of genres:", data[TARGET].nunique())
print("\nSample prepared data:")
print(data.head())
