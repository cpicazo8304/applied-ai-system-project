
import csv
from typing import List, Dict, Tuple

from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# ---------------------------------------------------------------------------
# Core functional pipeline
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """
    Parse a CSV file of songs and return a list of typed dicts.

    Each row is converted to a dict with proper Python types:
        - energy, valence, danceability, acousticness, tempo, loudness, liveness, speechiness → float
        - title, artist, genre, id  → str (whitespace stripped)

    Parameters
    ----------
    csv_path : str
        Path to the CSV file (e.g. "data/songs.csv").

    Returns
    -------
    List[Dict] — one dict per song row.

    Side Effects
    ------------
    Prints the number of loaded songs to stdout.
    """
    FLOAT_FIELDS = {"energy", "valence", "danceability", "acousticness", "loudness", "liveness", "speechiness", "tempo"}

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song: Dict = {}
            for key, val in row.items():
                if key in FLOAT_FIELDS or "cluster" in key:
                    song[key] = float(val)
                else:
                    song[key] = val.strip()
            songs.append(song)

    print(f"Loaded {len(songs)} songs from '{csv_path}'.")
    return songs

if __name__ == '__main__':
    import joblib
    import numpy as np

    # load the model
    gmm = joblib.load("./models/gmm_model.joblib")
    scaler = joblib.load("./models/scaler.joblib")

    # get user preferences as array in same feature order used during training
    # "danceability", "energy", "loudness", "speechiness", "acousticness", "liveness", "valence", "tempo"
    user_features = np.array([[
        0.5,  # danceability
        0.7,  # energy
        -5.0, # loudness
        0.05, # speechiness
        0.1,  # acousticness
        0.2,  # liveness
        0.6,  # valence
        120.0 # tempo
    ]])

    # scale using same scaler from training
    user_features_scaled = scaler.transform(user_features)

    # get user's cluster probability vector
    user_proba = gmm.predict_proba(user_features_scaled)[0]  # shape (n_components,)
    print("User's cluster probabilities:", user_proba)

    songs = load_songs("../data/songs_with_clusters.csv")

    # # For each song, compute cosine similarity between user's cluster vector and song's cluster vector
    # songs_proba = 
    # song_similarities = []
    # for song in songs:
    #     song_proba = np.array([song[f"cluster_{i}"] for i in range(len(user_proba))])  # shape (n_components,)
    #     sim = cosine_similarity(user_proba, song_proba)
    #     song_similarities.append((song["name"], sim))
    
    songs_proba = np.array([[song[f"cluster_{i}"] for i in range(len(user_proba))] for song in songs])  # shape (n_songs, n_components)
    similarities = np.dot(songs_proba, user_proba) / (norm(songs_proba, axis=1) * norm(user_proba))

    # get top 300 candidate indices
    top_indices = np.argsort(similarities)[::-1][:300]
    top_songs = [songs[i] for i in top_indices]
    print("Top 10 recommended songs:")
    for song in top_songs[:10]:
        print(f"{song['name']} by {song['artists']} (similarity: {similarities[songs.index(song)]:.4f})")