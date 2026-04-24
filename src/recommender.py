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
import anthropic
from typing import List, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "env", ".env"))

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Song:
    """
    Represents a song and its audio/metadata attributes.
    Required by tests/test_recommender.py.

    Attributes
    ----------
    id           : Unique string identifier (Spotify track ID).
    title        : Song title.
    artist       : Artist name.
    genre        : Genre label (must exist in GENRE_SIMILARITY).
    energy       : Perceived energy level [0, 1].
    tempo_bpm    : Tempo in beats per minute.
    valence      : Musical positivity [0, 1].
    danceability : How suitable for dancing [0, 1].
    acousticness : Degree of acoustic instrumentation [0, 1].
    loudness     : Overall loudness in decibels.
    liveness     : Presence of live audience [0, 1].
    speechiness  : Presence of spoken words [0, 1].
    """
    id: str
    title: str
    artist: str
    genre: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    loudness: float
    liveness: float
    speechiness: float


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


    def like(self, song: Song):
        """
        Update the user profile based on a liked song.
        This method can be called after a recommendation is made and the user indicates they like the song.
        The profile should be adjusted to increase the weights of attributes similar to the liked song.
        """
        self.num_interactions += 1
        self.log.append(f"Liked song: {song.title} by {song.artist}")

        # update preferred energy, acousticness, valence, tempo, danceability towards the liked song's attributes
        old_energy = self.preferred_energy
        old_valence = self.preferred_valence
        old_acousticness = self.preferred_acousticness
        old_tempo = self.preferred_tempo
        old_danceability = self.preferred_danceability
        old_speechiness = self.preferred_speechiness
        old_loudness = self.preferred_loudness
        old_liveness = self.preferred_liveness

        self.preferred_energy = self.alpha * song.energy + (1 - self.alpha) * self.preferred_energy
        self.preferred_acousticness = self.alpha * song.acousticness + (1 - self.alpha) * self.preferred_acousticness
        self.preferred_valence = self.alpha * song.valence + (1 - self.alpha) * self.preferred_valence
        self.preferred_tempo = self.alpha * song.tempo_bpm + (1 - self.alpha) * self.preferred_tempo
        self.preferred_danceability = self.alpha * song.danceability + (1 - self.alpha) * self.preferred_danceability
        self.preferred_speechiness = self.alpha * song.speechiness + (1 - self.alpha) * self.preferred_speechiness
        self.preferred_loudness = self.alpha * song.loudness + (1 - self.alpha) * self.preferred_loudness
        self.preferred_liveness = self.alpha * song.liveness + (1 - self.alpha) * self.preferred_liveness

        self.log.append(f"Liked '{song.title}' by {song.artist} → "
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
            

    def skip(self, song: Song):
        """
        Update the user profile based on a skipped song.
        This method can be called after a recommendation is made and the user indicates they skip the song.
        The profile should be adjusted to decrease the weights of attributes similar to the skipped song.
        """
        self.num_interactions += 1
        self.log.append(f"Skipped song: {song.title} by {song.artist}")

        old_energy = self.preferred_energy
        old_valence = self.preferred_valence
        old_acousticness = self.preferred_acousticness
        old_tempo = self.preferred_tempo
        old_danceability = self.preferred_danceability
        old_speechiness = self.preferred_speechiness
        old_loudness = self.preferred_loudness
        old_liveness = self.preferred_liveness

        self.preferred_energy = (1 + self.beta) * self.preferred_energy - self.beta * song.energy
        self.preferred_acousticness = (1 + self.beta) * self.preferred_acousticness - self.beta * song.acousticness
        self.preferred_valence = (1 + self.beta) * self.preferred_valence - self.beta * song.valence
        self.preferred_tempo = (1 + self.beta) * self.preferred_tempo - self.beta * song.tempo_bpm
        self.preferred_danceability = (1 + self.beta) * self.preferred_danceability - self.beta * song.danceability
        self.preferred_speechiness = (1 + self.beta) * self.preferred_speechiness - self.beta * song.speechiness
        self.preferred_loudness = (1 + self.beta) * self.preferred_loudness - self.beta * song.loudness
        self.preferred_liveness = (1 + self.beta) * self.preferred_liveness - self.beta * song.liveness

        self.log.append(f"Skipped '{song.title}' by {song.artist} → "
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
                {"energy": float, "acousticness": float, "mood": float, "valence": float, "tempo": float, 
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

# 16x16 fuzzy similarity matrix for genres.
GENRE_SIMILARITY: Dict[str, Dict[str, float]] = {
    "ambient":    {"ambient":1.0,"classical":0.5,"country":0.3,"edm":0.5,"electronic":0.6,"folk":0.4,"gospel":0.3,"hip-hop":0.2,"indie pop":0.3,"jazz":0.4,"lofi":0.8,"metal":0.1,"pop":0.2,"reggae":0.3,"rock":0.2,"synthwave":0.7,"world":0.4},
    "classical":  {"ambient":0.5,"classical":1.0,"country":0.4,"edm":0.2,"electronic":0.3,"folk":0.6,"gospel":0.7,"hip-hop":0.1,"indie pop":0.5,"jazz":0.8,"lofi":0.4,"metal":0.2,"pop":0.4,"reggae":0.3,"rock":0.3,"synthwave":0.2,"world":0.5},
    "country":    {"ambient":0.3,"classical":0.4,"country":1.0,"edm":0.2,"electronic":0.2,"folk":0.8,"gospel":0.5,"hip-hop":0.3,"indie pop":0.6,"jazz":0.4,"lofi":0.3,"metal":0.4,"pop":0.5,"reggae":0.4,"rock":0.7,"synthwave":0.1,"world":0.4},
    "edm":        {"ambient":0.5,"classical":0.2,"country":0.2,"edm":1.0,"electronic":0.9,"folk":0.2,"gospel":0.3,"hip-hop":0.6,"indie pop":0.4,"jazz":0.2,"lofi":0.4,"metal":0.3,"pop":0.7,"reggae":0.4,"rock":0.4,"synthwave":0.8,"world":0.3},
    "electronic": {"ambient":0.6,"classical":0.3,"country":0.2,"edm":0.9,"electronic":1.0,"folk":0.3,"gospel":0.3,"hip-hop":0.5,"indie pop":0.4,"jazz":0.3,"lofi":0.5,"metal":0.3,"pop":0.6,"reggae":0.4,"rock":0.4,"synthwave":0.8,"world":0.4},
    "folk":       {"ambient":0.4,"classical":0.6,"country":0.8,"edm":0.2,"electronic":0.3,"folk":1.0,"gospel":0.5,"hip-hop":0.2,"indie pop":0.7,"jazz":0.5,"lofi":0.5,"metal":0.3,"pop":0.5,"reggae":0.4,"rock":0.6,"synthwave":0.2,"world":0.5},
    "gospel":     {"ambient":0.3,"classical":0.7,"country":0.5,"edm":0.3,"electronic":0.3,"folk":0.5,"gospel":1.0,"hip-hop":0.4,"indie pop":0.5,"jazz":0.8,"lofi":0.3,"metal":0.2,"pop":0.6,"reggae":0.5,"rock":0.4,"synthwave":0.2,"world":0.7},
    "hip-hop":    {"ambient":0.2,"classical":0.1,"country":0.3,"edm":0.6,"electronic":0.5,"folk":0.2,"gospel":0.4,"hip-hop":1.0,"indie pop":0.5,"jazz":0.3,"lofi":0.3,"metal":0.4,"pop":0.8,"reggae":0.7,"rock":0.5,"synthwave":0.4,"world":0.5},
    "indie pop":  {"ambient":0.3,"classical":0.5,"country":0.6,"edm":0.4,"electronic":0.4,"folk":0.7,"gospel":0.5,"hip-hop":0.5,"indie pop":1.0,"jazz":0.6,"lofi":0.4,"metal":0.5,"pop":0.9,"reggae":0.4,"rock":0.7,"synthwave":0.4,"world":0.4},
    "jazz":       {"ambient":0.4,"classical":0.8,"country":0.4,"edm":0.2,"electronic":0.3,"folk":0.5,"gospel":0.8,"hip-hop":0.3,"indie pop":0.6,"jazz":1.0,"lofi":0.5,"metal":0.2,"pop":0.5,"reggae":0.4,"rock":0.4,"synthwave":0.3,"world":0.6},
    "lofi":       {"ambient":0.8,"classical":0.4,"country":0.3,"edm":0.4,"electronic":0.5,"folk":0.5,"gospel":0.3,"hip-hop":0.3,"indie pop":0.4,"jazz":0.5,"lofi":1.0,"metal":0.2,"pop":0.3,"reggae":0.3,"rock":0.3,"synthwave":0.5,"world":0.4},
    "metal":      {"ambient":0.1,"classical":0.2,"country":0.4,"edm":0.3,"electronic":0.3,"folk":0.3,"gospel":0.2,"hip-hop":0.4,"indie pop":0.5,"jazz":0.2,"lofi":0.2,"metal":1.0,"pop":0.4,"reggae":0.2,"rock":0.9,"synthwave":0.3,"world":0.3},
    "pop":        {"ambient":0.2,"classical":0.4,"country":0.5,"edm":0.7,"electronic":0.6,"folk":0.5,"gospel":0.6,"hip-hop":0.8,"indie pop":0.9,"jazz":0.5,"lofi":0.3,"metal":0.4,"pop":1.0,"reggae":0.5,"rock":0.6,"synthwave":0.6,"world":0.4},
    "reggae":     {"ambient":0.3,"classical":0.3,"country":0.4,"edm":0.4,"electronic":0.4,"folk":0.4,"gospel":0.5,"hip-hop":0.7,"indie pop":0.4,"jazz":0.4,"lofi":0.3,"metal":0.2,"pop":0.5,"reggae":1.0,"rock":0.3,"synthwave":0.3,"world":0.8},
    "rock":       {"ambient":0.2,"classical":0.3,"country":0.7,"edm":0.4,"electronic":0.4,"folk":0.6,"gospel":0.4,"hip-hop":0.5,"indie pop":0.7,"jazz":0.4,"lofi":0.3,"metal":0.9,"pop":0.6,"reggae":0.3,"rock":1.0,"synthwave":0.4,"world":0.4},
    "synthwave":  {"ambient":0.7,"classical":0.2,"country":0.1,"edm":0.8,"electronic":0.8,"folk":0.2,"gospel":0.2,"hip-hop":0.4,"indie pop":0.4,"jazz":0.3,"lofi":0.5,"metal":0.3,"pop":0.6,"reggae":0.3,"rock":0.4,"synthwave":1.0,"world":0.3},
    "world":      {"ambient":0.4,"classical":0.5,"country":0.4,"edm":0.3,"electronic":0.4,"folk":0.5,"gospel":0.7,"hip-hop":0.5,"indie pop":0.4,"jazz":0.6,"lofi":0.4,"metal":0.3,"pop":0.4,"reggae":0.8,"rock":0.4,"synthwave":0.3,"world":1.0},
}

# # 16x16 fuzzy similarity matrix for moods.
# # Values in [0, 1] — 1.0 means identical, 0.0 means completely unrelated.
# MOOD_SIMILARITY: Dict[str, Dict[str, float]] = {
#     "chill":       {"chill":1.0,"energetic":0.4,"exhilarated":0.3,"focused":0.8,"happy":0.5,"hopeful":0.6,"inspired":0.5,"intense":0.3,"laid-back":0.9,"melancholic":0.6,"moody":0.7,"nostalgic":0.7,"peaceful":0.9,"playful":0.4,"relaxed":0.9,"tribal":0.4},
#     "energetic":   {"chill":0.4,"energetic":1.0,"exhilarated":0.8,"focused":0.6,"happy":0.7,"hopeful":0.7,"inspired":0.6,"intense":0.9,"laid-back":0.5,"melancholic":0.3,"moody":0.4,"nostalgic":0.4,"peaceful":0.4,"playful":0.8,"relaxed":0.5,"tribal":0.6},
#     "exhilarated": {"chill":0.3,"energetic":0.8,"exhilarated":1.0,"focused":0.5,"happy":0.8,"hopeful":0.7,"inspired":0.7,"intense":0.9,"laid-back":0.4,"melancholic":0.2,"moody":0.3,"nostalgic":0.3,"peaceful":0.3,"playful":0.9,"relaxed":0.4,"tribal":0.5},
#     "focused":     {"chill":0.8,"energetic":0.6,"exhilarated":0.5,"focused":1.0,"happy":0.4,"hopeful":0.6,"inspired":0.6,"intense":0.5,"laid-back":0.7,"melancholic":0.5,"moody":0.6,"nostalgic":0.6,"peaceful":0.7,"playful":0.4,"relaxed":0.7,"tribal":0.4},
#     "happy":       {"chill":0.5,"energetic":0.7,"exhilarated":0.8,"focused":0.4,"happy":1.0,"hopeful":0.8,"inspired":0.8,"intense":0.6,"laid-back":0.6,"melancholic":0.3,"moody":0.4,"nostalgic":0.5,"peaceful":0.5,"playful":0.9,"relaxed":0.6,"tribal":0.5},
#     "hopeful":     {"chill":0.6,"energetic":0.7,"exhilarated":0.7,"focused":0.6,"happy":0.8,"hopeful":1.0,"inspired":0.9,"intense":0.5,"laid-back":0.6,"melancholic":0.4,"moody":0.5,"nostalgic":0.7,"peaceful":0.6,"playful":0.7,"relaxed":0.7,"tribal":0.5},
#     "inspired":    {"chill":0.5,"energetic":0.6,"exhilarated":0.7,"focused":0.6,"happy":0.8,"hopeful":0.9,"inspired":1.0,"intense":0.5,"laid-back":0.5,"melancholic":0.4,"moody":0.5,"nostalgic":0.8,"peaceful":0.5,"playful":0.7,"relaxed":0.6,"tribal":0.6},
#     "intense":     {"chill":0.3,"energetic":0.9,"exhilarated":0.9,"focused":0.5,"happy":0.6,"hopeful":0.5,"inspired":0.5,"intense":1.0,"laid-back":0.4,"melancholic":0.7,"moody":0.8,"nostalgic":0.4,"peaceful":0.3,"playful":0.6,"relaxed":0.4,"tribal":0.7},
#     "laid-back":   {"chill":0.9,"energetic":0.5,"exhilarated":0.4,"focused":0.7,"happy":0.6,"hopeful":0.6,"inspired":0.5,"intense":0.4,"laid-back":1.0,"melancholic":0.5,"moody":0.6,"nostalgic":0.6,"peaceful":0.8,"playful":0.5,"relaxed":0.9,"tribal":0.5},
#     "melancholic": {"chill":0.6,"energetic":0.3,"exhilarated":0.2,"focused":0.5,"happy":0.3,"hopeful":0.4,"inspired":0.4,"intense":0.7,"laid-back":0.5,"melancholic":1.0,"moody":0.9,"nostalgic":0.7,"peaceful":0.6,"playful":0.2,"relaxed":0.5,"tribal":0.4},
#     "moody":       {"chill":0.7,"energetic":0.4,"exhilarated":0.3,"focused":0.6,"happy":0.4,"hopeful":0.5,"inspired":0.5,"intense":0.8,"laid-back":0.6,"melancholic":0.9,"moody":1.0,"nostalgic":0.6,"peaceful":0.6,"playful":0.3,"relaxed":0.6,"tribal":0.5},
#     "nostalgic":   {"chill":0.7,"energetic":0.4,"exhilarated":0.3,"focused":0.6,"happy":0.5,"hopeful":0.7,"inspired":0.8,"intense":0.4,"laid-back":0.6,"melancholic":0.7,"moody":0.6,"nostalgic":1.0,"peaceful":0.7,"playful":0.4,"relaxed":0.7,"tribal":0.5},
#     "peaceful":    {"chill":0.9,"energetic":0.4,"exhilarated":0.3,"focused":0.7,"happy":0.5,"hopeful":0.6,"inspired":0.5,"intense":0.3,"laid-back":0.8,"melancholic":0.6,"moody":0.6,"nostalgic":0.7,"peaceful":1.0,"playful":0.4,"relaxed":0.9,"tribal":0.4},
#     "playful":     {"chill":0.4,"energetic":0.8,"exhilarated":0.9,"focused":0.4,"happy":0.9,"hopeful":0.7,"inspired":0.7,"intense":0.6,"laid-back":0.5,"melancholic":0.2,"moody":0.3,"nostalgic":0.4,"peaceful":0.4,"playful":1.0,"relaxed":0.5,"tribal":0.6},
#     "relaxed":     {"chill":0.9,"energetic":0.5,"exhilarated":0.4,"focused":0.7,"happy":0.6,"hopeful":0.7,"inspired":0.6,"intense":0.4,"laid-back":0.9,"melancholic":0.5,"moody":0.6,"nostalgic":0.7,"peaceful":0.9,"playful":0.5,"relaxed":1.0,"tribal":0.5},
#     "tribal":      {"chill":0.4,"energetic":0.6,"exhilarated":0.5,"focused":0.4,"happy":0.5,"hopeful":0.5,"inspired":0.6,"intense":0.7,"laid-back":0.5,"melancholic":0.4,"moody":0.5,"nostalgic":0.5,"peaceful":0.4,"playful":0.6,"relaxed":0.5,"tribal":1.0},
# }


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

    def __init__(self, songs: List[Song]):
        """
        Parameters
        ----------
        songs : List[Song]
            Full song catalog as Song dataclass instances.
        """
        self.songs = songs


    def recommend(self, user: UserProfile, k: int = 5) -> List[Tuple[Song, float, str]]:
        """
        Return the top-k songs best matching the user profile.

        Parameters
        ----------
        user : UserProfile
            The user's listening preferences.
        k : int
            Number of recommendations to return (default 5).

        Returns
        -------
        List of (Song, score, explanation) tuples sorted by score descending.
        """
        song_dicts = [vars(s) for s in self.songs]
        results = self.recommend_songs(user, song_dicts, k)
        id_to_song = {s.id: s for s in self.songs}
        return [(id_to_song[r[0]["id"]], r[1], r[2]) for r in results]
    

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
        tempo_sim = max(0.0, min(1.0, 1.0 - abs(song["tempo_bpm"] - prefs["preferred_tempo"]) / (TEMPO_MAX - TEMPO_MIN)))
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
        
        return
    
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
        scored = [(song, *self.score_song(user_prefs, song)) for song in songs]
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


# def _mood_sim(m1: str, m2: str) -> float:
#     """Return fuzzy mood similarity between m1 and m2 (defaults to 0.0 if unknown)."""
#     return MOOD_SIMILARITY.get(m1, {}).get(m2, 0.0)


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
                if key in FLOAT_FIELDS:
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

    recommender = Recommender([Song(**s) for s in songs])

    print("\nTop 5 Recommendations:")
    print("-" * 50)
    for song, score, explanation in recommender.recommend_songs(user_prefs, songs, k=5):
        print(f"[{score:.4f}] {song['title']} by {song['artist']} ({song['genre']})")
        print(explanation)
        print("-" * 50)