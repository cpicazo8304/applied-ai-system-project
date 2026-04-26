"""
recommender.py
==============
Core logic for the Music Recommender System.

Pipeline
--------
1. load_songs()        – Parse data/songs.csv into typed dicts.
2. score_song()        – Score a single song against a user profile using a
                         weighted similarity formula.
3. recommend_songs()   – Score every song, rank by score, return top-k.

Scoring weights (must sum to 1.0)
----------------------------------
    "energy":        0.22,
    "acousticness":  0.18,
    "valence":        0.13,
    "tempo":          0.10,
    "danceability":   0.08,
    "loudness":       0.08,
    "liveness":       0.06,
    "speechiness":    0.05,
    "genre":          0.06,
    "artist":         0.02,
    "title":          0.02

Similarity formulas
-------------------
- Numerical (energy, acousticness, valence, danceability, loudness, liveness, speechiness):
      sim = 1 - |song_val - pref_val|       (clamped to [0, 1])
- Tempo (BPM):
      sim = 1 - |song_bpm - pref_bpm| / (MAX_BPM - MIN_BPM)
- Genre:
      fuzzy lookup via pre-defined similarity matrices (see below)
- Artist / Title:
      exact match → 1.0, otherwise 0.0
"""

import os
import csv
import json
import joblib
import anthropic
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "env", ".env"))

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


class UserProfile:
    """
    Represents a user's listening preferences for the OOP interface.
    Required by tests/test_recommender.py.
    """

    def __init__(self, favorite_genres: List[str], favorite_artists: List[str]):
        self.favorite_genres = favorite_genres
        self.favorite_artists = favorite_artists
        # given favorite artists, favorite genres, initialize a profile for the user
        prompt = f"""
        You are a music recommendation system. Given a user's favorite artists and genres, create a 
        user profile with the following attributes:
        - preferred_energy: A float in [0, 1] representing the user's preferred energy level.
        - preferred_acousticness: A float in [0, 1] representing the user's preference for acoustic songs.
        - preferred_valence: A float in [0, 1] representing the user's preference for musical positivity.
        - preferred_tempo: A float representing the user's preferred tempo in BPM.
        - preferred_danceability: A float in [0, 1] representing the user's preference for danceable songs.
        - preferred_speechiness: A float in [0, 1] representing the user's preference for spoken word content in songs.
        - preferred_loudness: A float representing the user's preferred loudness level in decibels.
        - preferred_liveness: A float in [0, 1] representing the user's preference for live performance feel in songs.
        Here is the user's favorite artists: {', '.join(favorite_artists)}.
        Here is the user's favorite genres: {', '.join(favorite_genres)}.
        Return a JSON object with the profile attributes and their values, with no additional text or 
        explanation or preamble:
        {
            {
                "preferred_energy": float,
                "preferred_acousticness": float,
                "preferred_valence": float,
                "preferred_tempo": float,
                "preferred_danceability": float,
                "preferred_speechiness": float,
                "preferred_loudness": float,
                "preferred_liveness": float
            }
        }
        """
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        profile_data = json.loads(response.content[0].text)
        self.preferred_energy = profile_data.get("preferred_energy", 0.5)
        self.preferred_acousticness = profile_data.get("preferred_acousticness", 0.5)
        self.preferred_valence = profile_data.get("preferred_valence", 0.5)
        self.preferred_tempo = profile_data.get("preferred_tempo", 120.0)
        self.preferred_danceability = profile_data.get("preferred_danceability", 0.5)
        self.preferred_speechiness= profile_data.get("preferred_speechiness", 0.5)
        self.preferred_loudness = profile_data.get("preferred_loudness", 30)
        self.preferred_liveness = profile_data.get("preferred_liveness", 0.5)
        self.weights = {
            "energy":        0.22,
            "acousticness":  0.18,
            "valence":        0.13,
            "tempo":          0.10,
            "danceability":   0.08,
            "loudness":       0.08,
            "liveness":       0.06,
            "speechiness":    0.05,
            "genre":          0.06,
            "artist":         0.02,
            "title":          0.02
        }
        self.num_interactions = 0
        self.log = []
        self.ranked_songs = []
        self.alpha = 0.1
        self.beta = 0.1


    def add_genre_preference(self, genre: str):
        """
        Add a genre to the user's preferred genres list.
        This method can be called when the user indicates they like a song of a certain genre, to increase the weight of that genre in future recommendations.
        """
        if genre not in self.favorite_genres:
            self.favorite_genres.append(genre)

    
    def add_artist_preference(self, artist: str):
        """
        Add an artist to the user's preferred artists list.
        This method can be called when the user indicates they like a song by a certain artist, to increase the weight of that artist in future recommendations.
        """
        if artist not in self.favorite_artists:
            self.favorite_artists.append(artist)


    def like(self, song: Dict):
        """
        Update the user profile based on a liked song.
        This method can be called after a recommendation is made and the user indicates they like the song.
        The profile should be adjusted to increase the weights of attributes similar to the liked song.
        """
        self.num_interactions += 1
        self.log.append(f"Liked song: {song['title']} by {song['artist']}")

        # update preferred energy, acousticness, valence, tempo, danceability towards the liked song's attributes
        old_energy = self.preferred_energy
        old_valence = self.preferred_valence
        old_acousticness = self.preferred_acousticness
        old_tempo = self.preferred_tempo
        old_danceability = self.preferred_danceability
        old_speechiness = self.preferred_speechiness
        old_loudness = self.preferred_loudness
        old_liveness = self.preferred_liveness

        self.preferred_energy = self.alpha * song['energy'] + (1 - self.alpha) * self.preferred_energy
        self.preferred_acousticness = self.alpha * song['acousticness'] + (1 - self.alpha) * self.preferred_acousticness
        self.preferred_valence = self.alpha * song['valence'] + (1 - self.alpha) * self.preferred_valence
        self.preferred_tempo = self.alpha * song['tempo'] + (1 - self.alpha) * self.preferred_tempo
        self.preferred_danceability = self.alpha * song['danceability'] + (1 - self.alpha) * self.preferred_danceability
        self.preferred_speechiness = self.alpha * song['speechiness'] + (1 - self.alpha) * self.preferred_speechiness
        self.preferred_loudness = self.alpha * song['loudness'] + (1 - self.alpha) * self.preferred_loudness
        self.preferred_liveness = self.alpha * song['liveness'] + (1 - self.alpha) * self.preferred_liveness

        self.log.append(f"Liked '{song['title']}' by {song['artist']} → "
            f"energy: {old_energy:.2f} → {self.preferred_energy:.2f}, "
            f"acousticness: {old_acousticness:.2f} → {self.preferred_acousticness:.2f}, "
            f"valence: {old_valence:.2f} → {self.preferred_valence:.2f}, "
            f"tempo: {old_tempo:.2f} → {self.preferred_tempo:.2f}, "
            f"danceability: {old_danceability:.2f} → {self.preferred_danceability:.2f}, "
            f"loudness: {old_loudness:.2f} → {self.preferred_loudness:.2f}, "
            f"liveness: {old_liveness:.2f} → {self.preferred_liveness:.2f}, "
            f"speechiness: {old_speechiness:.2f} → {self.preferred_speechiness:.2f}")

        if abs(old_speechiness - self.preferred_speechiness) > 0.1 or abs(old_loudness - self.preferred_loudness) > 0.1 or abs(old_liveness - self.preferred_liveness) > 0.1 or abs(old_energy - self.preferred_energy) > 0.1 or abs(old_valence - self.preferred_valence) > 0.1 or abs(old_acousticness - self.preferred_acousticness) > 0.1 or abs(old_tempo - self.preferred_tempo) > 10 or abs(old_danceability - self.preferred_danceability) > 0.1:
            self.check_ranked_recommendations(self.ranked_songs)
            

    def skip(self, song: Dict):
        """
        Update the user profile based on a skipped song.
        This method can be called after a recommendation is made and the user indicates they skip the song.
        The profile should be adjusted to decrease the weights of attributes similar to the skipped song.
        """
        self.num_interactions += 1
        self.log.append(f"Skipped song: {song['title']} by {song['artist']}")

        old_energy = self.preferred_energy
        old_valence = self.preferred_valence
        old_acousticness = self.preferred_acousticness
        old_tempo = self.preferred_tempo
        old_danceability = self.preferred_danceability
        old_speechiness = self.preferred_speechiness
        old_loudness = self.preferred_loudness
        old_liveness = self.preferred_liveness

        self.preferred_energy = (1 + self.beta) * self.preferred_energy - self.beta * song['energy']
        self.preferred_acousticness = (1 + self.beta) * self.preferred_acousticness - self.beta * song['acousticness']
        self.preferred_valence = (1 + self.beta) * self.preferred_valence - self.beta * song['valence']
        self.preferred_tempo = (1 + self.beta) * self.preferred_tempo - self.beta * song['tempo']
        self.preferred_danceability = (1 + self.beta) * self.preferred_danceability - self.beta * song['danceability']
        self.preferred_speechiness = (1 + self.beta) * self.preferred_speechiness - self.beta * song['speechiness']
        self.preferred_loudness = (1 + self.beta) * self.preferred_loudness - self.beta * song['loudness']
        self.preferred_liveness = (1 + self.beta) * self.preferred_liveness - self.beta * song['liveness']

        self.log.append(f"Skipped '{song['title']}' by {song['artist']} → "
                        f"energy: {old_energy:.2f} → {self.preferred_energy:.2f}, "
                        f"acousticness: {old_acousticness:.2f} → {self.preferred_acousticness:.2f}, "
                        f"valence: {old_valence:.2f} → {self.preferred_valence:.2f}, "
                        f"tempo: {old_tempo:.2f} → {self.preferred_tempo:.2f}, "
                        f"danceability: {old_danceability:.2f} → {self.preferred_danceability:.2f}, "
                        f"loudness: {old_loudness:.2f} → {self.preferred_loudness:.2f}, "
                        f"liveness: {old_liveness:.2f} → {self.preferred_liveness:.2f}, "
                        f"speechiness: {old_speechiness:.2f} → {self.preferred_speechiness:.2f}")

        if (abs(old_energy - self.preferred_energy) > 0.1 or
            abs(old_valence - self.preferred_valence) > 0.1 or
            abs(old_acousticness - self.preferred_acousticness) > 0.1 or
            abs(old_tempo - self.preferred_tempo) > 10 or
            abs(old_danceability - self.preferred_danceability) > 0.1 or
            abs(old_speechiness - self.preferred_speechiness) > 0.1 or
            abs(old_loudness - self.preferred_loudness) > 0.1 or
            abs(old_liveness - self.preferred_liveness) > 0.1):
            self.check_ranked_recommendations(self.ranked_songs)

    
    def get_weights(self):
        """
        Return the current feature weights in the user profile.
        This method can be used to inspect the importance of different features when scoring songs.
        """
        return self.weights

    def update_weights(self, new_weights):
        """
        Update the feature weights in the user profile.
        This method can be used to adjust the importance of different features based on user feedback or changing preferences.
        new_weights should be a dict with the same keys as self.weights and values in [0, 1] that sum to 1.0.
        """
        self.weights = new_weights

    def structure_profile(self):
        """
        Return the user profile as a structured dict that can be used for scoring songs.
        This method converts the user profile attributes into the format expected by the scoring function.
        """
        return {
            "preferred_energy": self.preferred_energy,
            "preferred_acousticness": self.preferred_acousticness,
            "preferred_valence": self.preferred_valence,
            "preferred_tempo": self.preferred_tempo,
            "preferred_danceability": self.preferred_danceability,
            "preferred_speechiness": self.preferred_speechiness,
            "preferred_loudness": self.preferred_loudness,
            "preferred_liveness": self.preferred_liveness,
            "preferred_genres": {genre: 1.0 for genre in self.favorite_genres},
            "favorite_artist": self.favorite_artists[0] if self.favorite_artists else "",
            "favorite_title": "",
        }

    def update_ranked_songs(self, ranked_songs: List[Tuple[Dict, float, str]]):
        """
        Update the ranked songs list in the user profile.
        This method can be called after scoring songs to store the current ranking of songs based on the user's preferences.
        ranked_songs should be a list of (song_dict, score, explanation) tuples sorted by score descending.
        """
        self.ranked_songs = ranked_songs

    def get_k_ranked_songs_ids_with_ranks(self, k: int):
        """
        Return the IDs of the top-k ranked songs based on the current user profile.
        This method can be used to retrieve the most relevant song IDs for the user at any point in time.
        It assumes that self.ranked_songs is a list of (song, score, explanation) tuples sorted by score descending.
        """
        return {
            song['id']: i + 1 for i, (song, _, _) in enumerate(self.ranked_songs[:k])
        }
    
    def get_k_ranked_songs(self, k: int):
        """
        Return the top-k ranked songs based on the current user profile.
        This method can be used to retrieve the most relevant songs for the user at any point in time.
        It assumes that self.ranked_songs is a list of (song, score, explanation) tuples sorted by score descending.
        """
        return {
            song['id']: (song, score, explanation) for song, score, explanation in self.ranked_songs[:k]
        }
    
    def get_k_ranked_explanations(self, k: int):
        """
        Return the explanations for the top-k ranked songs based on the current user profile.
        This method can be used to understand why certain songs are recommended to the user.
        It assumes that self.ranked_songs is a list of (song, score, explanation) tuples sorted by score descending.
        """
        return {
            song['id']: explanation for song, _, explanation in self.ranked_songs[:k]
        }
    

    def check_ranked_recommendations(self, recommendations: List[Tuple[Dict, float, str]]) -> None:
        """
        Use an LLM to check if the ranked recommendations make logical sense given the user's preferences.
        Flags any contradictions and produces a reliability score (0-1) for the overall recommendation set.
        Also generates plain English explanations for why each song was ranked where it was. If the reliability score 
        is below a certain threshold (e.g. 0.70), prompts the LLM to suggest adjustments to the scoring weights to 
        better suit the user's profile.
        """
        recommendations_structured = structure_recommendations_for_llm(recommendations)
        user_prefs_structured = self.structure_profile()
        prompt1 = f""""
        Given top 5 recommendations with the scores, and user profile, check if the rankings make 
        logical sense given the user's preferences. Flag any contradictions and produce a reliability 
        score (0-1) for the overall recommendation set such as a high energy song ranking highly for a 
        low energy user because tempo dominated.
        Here are the recommendations with scores: {recommendations_structured}. 
        Here is the user profile: {user_prefs_structured}.
        Respond in JSON format with no preamble: {
            {"reliability_score": float, "contradictions": List[str]}
        }.
        """

        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt1}
            ]
        )

        result = json.loads(response.content[0].text)
        reliability_score = result.get("reliability_score", 0.0)
        contradictions = result.get("contradictions", [])

        if reliability_score < 0.70:
            print(f"Warning: Low reliability score ({reliability_score:.2f}) for recommendations. Contradictions found:")
            for contradiction in contradictions:
                print(f" - {contradiction}")

            # change weights
            prompt2 = f"""Given a not good enough reliability score {reliability_score}, can you change the 
            given weights to better suit the current user? Here are the contradictions: {contradictions}. 
            Here are the current weights: {self.get_weights()}. Here is the user profile: {user_prefs_structured}. 
            Only adjust weights that are contributing to the low reliability score, leave others unchanged. 
            Weights must sum to 1.0. Respond only in this exact JSON format with no preamble: 
            {
                {"energy": float, "acousticness": float, "valence": float, "tempo": float, 
            "danceability": float, "loudness": float, "liveness": float, "speechiness": float, "genre": float, "artist": float, "title": float}
            }
            """
            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt2}
                ]
            )
            new_weights = json.loads(response.content[0].text)

            # Update weights with new values
            self.update_weights(new_weights)

            total = sum(self.get_weights().values())
            if abs(total - 1.0) > 0.01:
                print(f"Warning: adjusted weights sum to {total:.3f}, not 1.0")



# ---------------------------------------------------------------------------
# Similarity matrices
# ---------------------------------------------------------------------------

GENRE_SIMILARITY: Dict[str, Dict[str, float]] = {
    "Rock":       {"Rock":1.0, "Hip-Hop":0.5, "Classical":0.3, "Pop":0.6, "Jazz":0.4, "R&B":0.5, "Blues":0.6, "Electronic":0.4, "Country":0.7, "Folk":0.6},
    "Hip-Hop":    {"Rock":0.5, "Hip-Hop":1.0, "Classical":0.1, "Pop":0.8, "Jazz":0.3, "R&B":0.8, "Blues":0.4, "Electronic":0.5, "Country":0.3, "Folk":0.2},
    "Classical":  {"Rock":0.3, "Hip-Hop":0.1, "Classical":1.0, "Pop":0.4, "Jazz":0.8, "R&B":0.3, "Blues":0.5, "Electronic":0.3, "Country":0.4, "Folk":0.6},
    "Pop":        {"Rock":0.6, "Hip-Hop":0.8, "Classical":0.4, "Pop":1.0, "Jazz":0.5, "R&B":0.7, "Blues":0.4, "Electronic":0.6, "Country":0.5, "Folk":0.5},
    "Jazz":       {"Rock":0.4, "Hip-Hop":0.3, "Classical":0.8, "Pop":0.5, "Jazz":1.0, "R&B":0.6, "Blues":0.8, "Electronic":0.3, "Country":0.4, "Folk":0.5},
    "R&B":        {"Rock":0.5, "Hip-Hop":0.8, "Classical":0.3, "Pop":0.7, "Jazz":0.6, "R&B":1.0, "Blues":0.7, "Electronic":0.4, "Country":0.3, "Folk":0.3},
    "Blues":      {"Rock":0.6, "Hip-Hop":0.4, "Classical":0.5, "Pop":0.4, "Jazz":0.8, "R&B":0.7, "Blues":1.0, "Electronic":0.2, "Country":0.5, "Folk":0.6},
    "Electronic": {"Rock":0.4, "Hip-Hop":0.5, "Classical":0.3, "Pop":0.6, "Jazz":0.3, "R&B":0.4, "Blues":0.2, "Electronic":1.0, "Country":0.2, "Folk":0.3},
    "Country":    {"Rock":0.7, "Hip-Hop":0.3, "Classical":0.4, "Pop":0.5, "Jazz":0.4, "R&B":0.3, "Blues":0.5, "Electronic":0.2, "Country":1.0, "Folk":0.8},
    "Folk":       {"Rock":0.6, "Hip-Hop":0.2, "Classical":0.6, "Pop":0.5, "Jazz":0.5, "R&B":0.3, "Blues":0.6, "Electronic":0.3, "Country":0.8, "Folk":1.0},
}

# BPM bounds used for tempo normalization
TEMPO_MIN = 60.0
TEMPO_MAX = 200.0


# ---------------------------------------------------------------------------
# OOP interface
# ---------------------------------------------------------------------------

class Recommender:
    """
    Object-oriented interface for the music recommender.
    Wraps the functional pipeline (load_songs → score_song → recommend_songs)
    behind a stateful class that holds the song catalog.

    Required by tests/test_recommender.py.

    Usage
    -----
        songs = load_songs("data/songs.csv")
        song_objs = [Song(**s) for s in songs]
        rec = Recommender(song_objs)
        results = rec.recommend(user_profile, k=5)
        for song, score, explanation in results:
            print(song.title, score)
    """

    def __init__(self, songs: List[Dict]):
        """
        Parameters
        ----------
        songs : List[Dict]
            Full song catalog as dictionaries.
        """
        # load the model
        self.gmm = joblib.load("./models/gmm_model.joblib")
        self.scaler = joblib.load("./models/scaler.joblib")
        self.songs = songs
        self.songs_proba = np.array([[song[f"cluster_{i}"] for i in range(self.gmm.n_components)] for song in songs])  # shape (n_songs, n_components)


    def score_song(self, user_prefs: UserProfile, song: Dict) -> Tuple[float, List[str]]:
        """
        Score a single song against a user profile using the weighted
        similarity formula defined in the system spec.

        Parameters
        ----------
        user_prefs : UserProfile
            The user's listening preferences.
        song : Dict
            A song dict as returned by load_songs().
                preferred_genres  (Dict[str, float] — genre → weight),
                favorite_artist   (str),
                favorite_title    (str)
        song : Dict
            A song dict as returned by load_songs().

        Returns
        -------
        (score, reasons)
            score   : float in [0, 1] — overall weighted similarity.
            reasons : List[str] — one line per feature explaining its contribution.
        """
        reasons: List[str] = []

        def num_sim(song_val: float, pref_val: float, label: str, weight: float) -> float:
            """Compute and log a numerical feature's weighted contribution."""
            sim = max(0.0, min(1.0, 1.0 - abs(song_val - pref_val)))
            contrib = weight * sim
            reasons.append(f"{label}: sim={sim:.3f}, contrib={contrib:.4f} (w={weight})")
            return contrib

        total = 0.0
        weights = user_prefs.get_weights()
        prefs = user_prefs.structure_profile()

        # Numerical features
        total += num_sim(song["energy"],       prefs["preferred_energy"],       "energy",       weights["energy"])
        total += num_sim(song["acousticness"], prefs["preferred_acousticness"], "acousticness", weights["acousticness"])
        total += num_sim(song["valence"],      prefs["preferred_valence"],      "valence",      weights["valence"])
        total += num_sim(song["danceability"], prefs["preferred_danceability"], "danceability", weights["danceability"])
        total += num_sim(song["liveness"],     prefs["preferred_liveness"],     "liveness",     weights["liveness"])
        total += num_sim(song["speechiness"],  prefs["preferred_speechiness"],  "speechiness",  weights["speechiness"])

        LOUDNESS_MIN = -60.0
        LOUDNESS_MAX = 0.0

        loudness_sim = max(0.0, min(1.0, 1.0 - abs(song["loudness"] - prefs["preferred_loudness"]) / (LOUDNESS_MAX - LOUDNESS_MIN)))
        loudness_contrib = weights["loudness"] * loudness_sim
        reasons.append(f"loudness: sim={loudness_sim:.3f}, contrib={loudness_contrib:.4f} (w={weights['loudness']})")
        total += loudness_contrib

        # Tempo — normalised over BPM range
        tempo_sim = max(0.0, min(1.0, 1.0 - abs(song["tempo"] - prefs["preferred_tempo"]) / (TEMPO_MAX - TEMPO_MIN)))
        tempo_contrib = weights["tempo"] * tempo_sim
        reasons.append(f"tempo: sim={tempo_sim:.3f}, contrib={tempo_contrib:.4f} (w={weights['tempo']})")
        total += tempo_contrib

        # Genre — weighted average over preferred genres, normalised
        pref_genres: Dict[str, float] = prefs.get("preferred_genres", {})
        genre_sim = (
            sum(w * _genre_sim(song["genre"], g) for g, w in pref_genres.items()) / sum(pref_genres.values())
            if pref_genres else 0.0
        )
        genre_contrib = weights["genre"] * genre_sim
        reasons.append(f"genre ({song['genre']}): sim={genre_sim:.3f}, contrib={genre_contrib:.4f} (w={weights['genre']})")
        total += genre_contrib

        # Artist — exact match
        artist_sim = 1.0 if song["artist"] == prefs.get("favorite_artist", "") else 0.0
        artist_contrib = weights["artist"] * artist_sim
        reasons.append(
            f"artist match ({song['artist']}): contrib={artist_contrib:.4f} (w={weights['artist']})"
            if artist_sim else f"artist: no match, contrib=0.0000 (w={weights['artist']})"
        )
        total += artist_contrib

        # Title — exact match
        title_sim = 1.0 if song["title"] == prefs.get("favorite_title", "") else 0.0
        title_contrib = weights["title"] * title_sim
        reasons.append(
            f"title match ({song['title']}): contrib={title_contrib:.4f} (w={weights['title']})"
            if title_sim else f"title: no match, contrib=0.0000 (w={weights['title']})"
        )
        total += title_contrib

        return round(total, 6), reasons
    
    def get_candidates(self, user_prefs: UserProfile, songs: List[Dict]):
        """
        Get candidate songs for the user based on their preferences.
        This method uses the GMM model to find songs with similar feature distributions 
        to the user's preferences, returning a subset of songs to score in detail.
        """
        user_feats = np.array([
            user_prefs.preferred_danceability,
            user_prefs.preferred_energy,
            user_prefs.preferred_loudness,
            user_prefs.preferred_speechiness,
            user_prefs.preferred_acousticness,
            user_prefs.preferred_liveness,
            user_prefs.preferred_valence,
            user_prefs.preferred_tempo
        ]).reshape(1, -1)

        user_feats_scaled = self.scaler.transform(user_feats)
        user_proba = self.gmm.predict_proba(user_feats_scaled)[0]

        similarities = np.dot(self.songs_proba, user_proba) / (np.linalg.norm(self.songs_proba, axis=1) * np.linalg.norm(user_proba) + 1e-10)
        top_indices = np.argsort(similarities)[::-1][:100]
        candidates = [songs[i] for i in top_indices]

        return candidates
    
    def recommend_songs(self, user_prefs: UserProfile, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
        """
        Score every song in the catalog and return the top-k recommendations.

        Accepts either the simplified or full user_prefs format — normalisation
        is handled internally via _normalize_prefs().

        Parameters
        ----------
        user_prefs : Dict
            User preferences in simplified or full format (see _normalize_prefs).
        songs : List[Dict]
            Full song catalog as returned by load_songs().
        k : int
            Number of top results to return (default 5).

        Returns
        -------
        List of (song_dict, score, explanation) tuples, sorted by score descending.
        """
        prev_recommendation_ids = user_prefs.get_k_ranked_songs_ids_with_ranks(k)
        candidates = self.get_candidates(user_prefs, songs)
        scored = [(song, *self.score_song(user_prefs, song)) for song in candidates]
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        recommendations = [(song, score, "\n".join(reasons)) for song, score, reasons in ranked[:k]]

        # go through each ranked song and only keep the ones that are new 
        # or have changed position significantly (e.g. moved up by 3+ places) to generate explanations for
        new_recommendations = []
        for i, (song, score, reasons) in enumerate(recommendations):
            if song['id'] not in prev_recommendation_ids or (song['id'] in prev_recommendation_ids and abs(prev_recommendation_ids[song['id']] - (i + 1)) >= 3):
                # Generate explanation for this song
                new_recommendations.append((song, score, reasons))

        explanation_dict = self.get_explanations(user_prefs, new_recommendations)

        # give explanations to each of the top k songs, but reuse previous 
        # explanations for songs that are still in the top-k and haven't changed position 
        # significantly to save on LLM calls
        final_recommendations = []
        prev_explanations = user_prefs.get_k_ranked_explanations(k)
        for (song, score, _) in recommendations:
            # If the song is new or has changed position significantly, use the new explanation from the LLM.
            if song['id'] in explanation_dict:
                explanation = explanation_dict.get(song['id'], "No explanation available.")
            else:
                explanation = prev_explanations.get(song['id'], "No explanation available.")
            final_recommendations.append((song, score, explanation))

        return final_recommendations


    def get_explanations(self, user_prefs: UserProfile, recommendations: List[Tuple[Dict, float, str]]) -> Dict[str, str]:
        """
        Generate plain English explanations for why each song was ranked where it was.
        Uses an LLM to interpret the feature contributions and produce user-friendly explanations.

        Parameters
        ----------
        user_prefs : UserProfile
            The user's listening preferences.
        recommendations : List[Tuple[str, Tuple[Dict, float, str]]]
            A list of tuples mapping song IDs to (song_dict, score, explanation) tuples.

        Returns
        -------
        List of plain English explanations corresponding to each recommended song.
        """

        recommendations_structured = structure_recommendations_for_llm(recommendations)
        prompt = f"""Given the following ranked recommendations with their feature contributions, generate a plain English explanation for why each song was ranked where it was. 
        Focus on the most influential features and how they align or misalign with the user's preferences. 
        Here are the recommendations with scores and contributions: {recommendations_structured}. 
        Here is the user profile: {user_prefs.structure_profile()}.
        Respond with a JSON with no preamble, one per recommended song, in the 
        same order as the input recommendations: 
        {{"song_id": "explanation"}} with all recommended song IDs as keys and the generated explanations as values.
        """

        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        explanations = json.loads(response.content[0].text)
        return explanations

# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------
def _genre_sim(g1: str, g2: str) -> float:
        """Return fuzzy genre similarity between g1 and g2 (defaults to 0.0 if unknown)."""
        return GENRE_SIMILARITY.get(g1, {}).get(g2, 0.0)


def structure_recommendations_for_llm(recommendations: List[Tuple[Dict, float, str]]) -> str:
    """
    Convert the list of recommendations into a structured string format
    suitable for LLM input (e.g. for generating explanations or summaries).

    Parameters
    ----------
    recommendations : List of (song_dict, score, explanation) tuples.

    Returns
    -------
    A multi-line string with one section per recommended song, including:
        - Song title and artist
        - Overall similarity score
        - Per-feature contribution breakdown
    """
    lines = []
    for idx, (song, score, explanation) in enumerate(recommendations, start=1):
        lines.append(f"Recommendation #{idx}:")
        lines.append(f"ID: {song['id']}")
        lines.append(f"Title: {song['title']}")
        lines.append(f"Artist: {song['artist']}")
        lines.append(f"Genre: {song['genre']}")
        lines.append(f"Similarity Score: {score:.4f}")
        lines.append("Feature Contributions:")
        lines.extend(f"  - {line}" for line in explanation.split("\n"))
        lines.append("-" * 40)
    return "\n".join(lines)


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


# ---------------------------------------------------------------------------
# Manual test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    songs = load_songs(csv_path)

    user_prefs = UserProfile(
        favorite_genres=["jazz", "lofi"],
        favorite_artists=["Artist A", "Artist B"]
    )

    recommender = Recommender(songs)

    print("\nTop 5 Recommendations:")
    print("-" * 50)
    for song, score, explanation in recommender.recommend_songs(user_prefs, songs, k=5):
        print(f"[{score:.4f}] {song['title']} by {song['artist']} ({song['genre']})")
        print(explanation)
        print("-" * 50)