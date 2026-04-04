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
    energy        0.25
    acousticness  0.20
    mood          0.15
    valence       0.12
    tempo         0.10
    danceability  0.08
    genre         0.04
    artist        0.02
    title         0.02

Similarity formulas
-------------------
- Numerical (energy, acousticness, valence, danceability):
      sim = 1 - |song_val - pref_val|       (clamped to [0, 1])
- Tempo (BPM):
      sim = 1 - |song_bpm - pref_bpm| / (MAX_BPM - MIN_BPM)
- Genre / Mood:
      fuzzy lookup via pre-defined similarity matrices (see below)
- Artist / Title:
      exact match → 1.0, otherwise 0.0
"""

import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


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
    id           : Unique integer identifier.
    title        : Song title.
    artist       : Artist name.
    genre        : Genre label (must exist in GENRE_SIMILARITY).
    mood         : Mood label (must exist in MOOD_SIMILARITY).
    energy       : Perceived energy level [0, 1].
    tempo_bpm    : Tempo in beats per minute.
    valence      : Musical positivity [0, 1].
    danceability : How suitable for dancing [0, 1].
    acousticness : Degree of acoustic instrumentation [0, 1].
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    Represents a user's listening preferences for the OOP interface.
    Required by tests/test_recommender.py.

    Attributes
    ----------
    favorite_genre : User's preferred genre label.
    favorite_mood  : User's preferred mood label.
    target_energy  : Desired energy level [0, 1].
    likes_acoustic : True if the user prefers acoustic songs.
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


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

# 16x16 fuzzy similarity matrix for moods.
# Values in [0, 1] — 1.0 means identical, 0.0 means completely unrelated.
MOOD_SIMILARITY: Dict[str, Dict[str, float]] = {
    "chill":       {"chill":1.0,"energetic":0.4,"exhilarated":0.3,"focused":0.8,"happy":0.5,"hopeful":0.6,"inspired":0.5,"intense":0.3,"laid-back":0.9,"melancholic":0.6,"moody":0.7,"nostalgic":0.7,"peaceful":0.9,"playful":0.4,"relaxed":0.9,"tribal":0.4},
    "energetic":   {"chill":0.4,"energetic":1.0,"exhilarated":0.8,"focused":0.6,"happy":0.7,"hopeful":0.7,"inspired":0.6,"intense":0.9,"laid-back":0.5,"melancholic":0.3,"moody":0.4,"nostalgic":0.4,"peaceful":0.4,"playful":0.8,"relaxed":0.5,"tribal":0.6},
    "exhilarated": {"chill":0.3,"energetic":0.8,"exhilarated":1.0,"focused":0.5,"happy":0.8,"hopeful":0.7,"inspired":0.7,"intense":0.9,"laid-back":0.4,"melancholic":0.2,"moody":0.3,"nostalgic":0.3,"peaceful":0.3,"playful":0.9,"relaxed":0.4,"tribal":0.5},
    "focused":     {"chill":0.8,"energetic":0.6,"exhilarated":0.5,"focused":1.0,"happy":0.4,"hopeful":0.6,"inspired":0.6,"intense":0.5,"laid-back":0.7,"melancholic":0.5,"moody":0.6,"nostalgic":0.6,"peaceful":0.7,"playful":0.4,"relaxed":0.7,"tribal":0.4},
    "happy":       {"chill":0.5,"energetic":0.7,"exhilarated":0.8,"focused":0.4,"happy":1.0,"hopeful":0.8,"inspired":0.8,"intense":0.6,"laid-back":0.6,"melancholic":0.3,"moody":0.4,"nostalgic":0.5,"peaceful":0.5,"playful":0.9,"relaxed":0.6,"tribal":0.5},
    "hopeful":     {"chill":0.6,"energetic":0.7,"exhilarated":0.7,"focused":0.6,"happy":0.8,"hopeful":1.0,"inspired":0.9,"intense":0.5,"laid-back":0.6,"melancholic":0.4,"moody":0.5,"nostalgic":0.7,"peaceful":0.6,"playful":0.7,"relaxed":0.7,"tribal":0.5},
    "inspired":    {"chill":0.5,"energetic":0.6,"exhilarated":0.7,"focused":0.6,"happy":0.8,"hopeful":0.9,"inspired":1.0,"intense":0.5,"laid-back":0.5,"melancholic":0.4,"moody":0.5,"nostalgic":0.8,"peaceful":0.5,"playful":0.7,"relaxed":0.6,"tribal":0.6},
    "intense":     {"chill":0.3,"energetic":0.9,"exhilarated":0.9,"focused":0.5,"happy":0.6,"hopeful":0.5,"inspired":0.5,"intense":1.0,"laid-back":0.4,"melancholic":0.7,"moody":0.8,"nostalgic":0.4,"peaceful":0.3,"playful":0.6,"relaxed":0.4,"tribal":0.7},
    "laid-back":   {"chill":0.9,"energetic":0.5,"exhilarated":0.4,"focused":0.7,"happy":0.6,"hopeful":0.6,"inspired":0.5,"intense":0.4,"laid-back":1.0,"melancholic":0.5,"moody":0.6,"nostalgic":0.6,"peaceful":0.8,"playful":0.5,"relaxed":0.9,"tribal":0.5},
    "melancholic": {"chill":0.6,"energetic":0.3,"exhilarated":0.2,"focused":0.5,"happy":0.3,"hopeful":0.4,"inspired":0.4,"intense":0.7,"laid-back":0.5,"melancholic":1.0,"moody":0.9,"nostalgic":0.7,"peaceful":0.6,"playful":0.2,"relaxed":0.5,"tribal":0.4},
    "moody":       {"chill":0.7,"energetic":0.4,"exhilarated":0.3,"focused":0.6,"happy":0.4,"hopeful":0.5,"inspired":0.5,"intense":0.8,"laid-back":0.6,"melancholic":0.9,"moody":1.0,"nostalgic":0.6,"peaceful":0.6,"playful":0.3,"relaxed":0.6,"tribal":0.5},
    "nostalgic":   {"chill":0.7,"energetic":0.4,"exhilarated":0.3,"focused":0.6,"happy":0.5,"hopeful":0.7,"inspired":0.8,"intense":0.4,"laid-back":0.6,"melancholic":0.7,"moody":0.6,"nostalgic":1.0,"peaceful":0.7,"playful":0.4,"relaxed":0.7,"tribal":0.5},
    "peaceful":    {"chill":0.9,"energetic":0.4,"exhilarated":0.3,"focused":0.7,"happy":0.5,"hopeful":0.6,"inspired":0.5,"intense":0.3,"laid-back":0.8,"melancholic":0.6,"moody":0.6,"nostalgic":0.7,"peaceful":1.0,"playful":0.4,"relaxed":0.9,"tribal":0.4},
    "playful":     {"chill":0.4,"energetic":0.8,"exhilarated":0.9,"focused":0.4,"happy":0.9,"hopeful":0.7,"inspired":0.7,"intense":0.6,"laid-back":0.5,"melancholic":0.2,"moody":0.3,"nostalgic":0.4,"peaceful":0.4,"playful":1.0,"relaxed":0.5,"tribal":0.6},
    "relaxed":     {"chill":0.9,"energetic":0.5,"exhilarated":0.4,"focused":0.7,"happy":0.6,"hopeful":0.7,"inspired":0.6,"intense":0.4,"laid-back":0.9,"melancholic":0.5,"moody":0.6,"nostalgic":0.7,"peaceful":0.9,"playful":0.5,"relaxed":1.0,"tribal":0.5},
    "tribal":      {"chill":0.4,"energetic":0.6,"exhilarated":0.5,"focused":0.4,"happy":0.5,"hopeful":0.5,"inspired":0.6,"intense":0.7,"laid-back":0.5,"melancholic":0.4,"moody":0.5,"nostalgic":0.5,"peaceful":0.4,"playful":0.6,"relaxed":0.5,"tribal":1.0},
}

# Scoring weights — must sum to 1.0
WEIGHTS: Dict[str, float] = {
    "energy":       0.25,
    "acousticness": 0.20,
    "mood":         0.15,
    "valence":      0.12,
    "tempo":        0.10,
    "danceability": 0.08,
    "genre":        0.04,
    "artist":       0.02,
    "title":        0.02,
}

# BPM bounds used for tempo normalization
TEMPO_MIN = 60.0
TEMPO_MAX = 200.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _genre_sim(g1: str, g2: str) -> float:
    """Return fuzzy genre similarity between g1 and g2 (defaults to 0.0 if unknown)."""
    return GENRE_SIMILARITY.get(g1, {}).get(g2, 0.0)


def _mood_sim(m1: str, m2: str) -> float:
    """Return fuzzy mood similarity between m1 and m2 (defaults to 0.0 if unknown)."""
    return MOOD_SIMILARITY.get(m1, {}).get(m2, 0.0)


def _normalize_prefs(user_prefs: Dict) -> Dict:
    """
    Accepts either the simplified format used by main.py:
        {"genre": "pop", "mood": "happy", "energy": 0.8}
    or the full internal format:
        {"preferred_energy": 0.8, "preferred_genres": {...}, ...}

    Always returns the full format so score_song() can work with either.
    Missing numerical fields default to 0.5; missing artist/title default to "".
    """
    if "preferred_energy" in user_prefs:
        return user_prefs  # already full format, pass through

    genre = user_prefs.get("genre", "pop")
    mood  = user_prefs.get("mood", "chill")
    return {
        "preferred_energy":       user_prefs.get("energy", 0.5),
        "preferred_acousticness": user_prefs.get("acousticness", 0.5),
        "preferred_valence":      user_prefs.get("valence", 0.5),
        "preferred_tempo":        user_prefs.get("tempo", 120.0),
        "preferred_danceability": user_prefs.get("danceability", 0.5),
        "preferred_genres":       {genre: 1.0},
        "preferred_moods":        {mood: 1.0},
        "favorite_artist":        user_prefs.get("artist", ""),
        "favorite_title":         user_prefs.get("title", ""),
    }


def _profile_to_prefs(user: UserProfile) -> Dict:
    """
    Converts a UserProfile dataclass into the full user_prefs dict format
    expected by score_song(). Used internally by the Recommender class.
    """
    return {
        "preferred_energy":       user.target_energy,
        "preferred_acousticness": 0.8 if user.likes_acoustic else 0.2,
        "preferred_valence":      0.7,
        "preferred_tempo":        120.0,
        "preferred_danceability": 0.6,
        "preferred_genres":       {user.favorite_genre: 1.0},
        "preferred_moods":        {user.favorite_mood: 1.0},
        "favorite_artist":        "",
        "favorite_title":         "",
    }


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
        user_prefs = _profile_to_prefs(user)
        song_dicts = [vars(s) for s in self.songs]
        results = recommend_songs(user_prefs, song_dicts, k)
        id_to_song = {s.id: s for s in self.songs}
        return [(id_to_song[r[0]["id"]], r[1], r[2]) for r in results]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """
        Return a human-readable breakdown of why a song was recommended.

        Parameters
        ----------
        user : UserProfile
            The user's listening preferences.
        song : Song
            The song to explain.

        Returns
        -------
        A multi-line string with per-feature similarity and contribution scores.
        """
        user_prefs = _profile_to_prefs(user)
        _, reasons = score_song(user_prefs, vars(song))
        return "\n".join(reasons)


# ---------------------------------------------------------------------------
# Core functional pipeline
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """
    Parse a CSV file of songs and return a list of typed dicts.

    Each row is converted to a dict with proper Python types:
        - id           → int
        - energy, valence, danceability, acousticness, tempo_bpm → float
        - title, artist, genre, mood → str (whitespace stripped)

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
    FLOAT_FIELDS = {"energy", "valence", "danceability", "acousticness", "tempo_bpm"}
    INT_FIELDS   = {"id"}

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song: Dict = {}
            for key, val in row.items():
                if key in INT_FIELDS:
                    song[key] = int(val)
                elif key in FLOAT_FIELDS:
                    song[key] = float(val)
                else:
                    song[key] = val.strip()
            songs.append(song)

    print(f"Loaded {len(songs)} songs from '{csv_path}'.")
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Score a single song against a user profile using the weighted
    similarity formula defined in the system spec.

    Parameters
    ----------
    user_prefs : Dict
        Full preferences dict with keys:
            preferred_energy, preferred_acousticness, preferred_valence,
            preferred_tempo (BPM), preferred_danceability,
            preferred_genres  (Dict[str, float] — genre → weight),
            preferred_moods   (Dict[str, float] — mood  → weight),
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

    # Numerical features
    total += num_sim(song["energy"],       user_prefs["preferred_energy"],       "energy",       WEIGHTS["energy"])
    total += num_sim(song["acousticness"], user_prefs["preferred_acousticness"], "acousticness", WEIGHTS["acousticness"])
    total += num_sim(song["valence"],      user_prefs["preferred_valence"],      "valence",      WEIGHTS["valence"])
    total += num_sim(song["danceability"], user_prefs["preferred_danceability"], "danceability", WEIGHTS["danceability"])

    # Tempo — normalised over BPM range
    tempo_sim = max(0.0, min(1.0, 1.0 - abs(song["tempo_bpm"] - user_prefs["preferred_tempo"]) / (TEMPO_MAX - TEMPO_MIN)))
    tempo_contrib = WEIGHTS["tempo"] * tempo_sim
    reasons.append(f"tempo: sim={tempo_sim:.3f}, contrib={tempo_contrib:.4f} (w={WEIGHTS['tempo']})")
    total += tempo_contrib

    # Genre — weighted average over preferred genres, normalised
    pref_genres: Dict[str, float] = user_prefs.get("preferred_genres", {})
    genre_sim = (
        sum(w * _genre_sim(song["genre"], g) for g, w in pref_genres.items()) / sum(pref_genres.values())
        if pref_genres else 0.0
    )
    genre_contrib = WEIGHTS["genre"] * genre_sim
    reasons.append(f"genre ({song['genre']}): sim={genre_sim:.3f}, contrib={genre_contrib:.4f} (w={WEIGHTS['genre']})")
    total += genre_contrib

    # Mood — weighted average over preferred moods, normalised
    pref_moods: Dict[str, float] = user_prefs.get("preferred_moods", {})
    mood_sim = (
        sum(w * _mood_sim(song["mood"], m) for m, w in pref_moods.items()) / sum(pref_moods.values())
        if pref_moods else 0.0
    )
    mood_contrib = WEIGHTS["mood"] * mood_sim
    reasons.append(f"mood ({song['mood']}): sim={mood_sim:.3f}, contrib={mood_contrib:.4f} (w={WEIGHTS['mood']})")
    total += mood_contrib

    # Artist — exact match
    artist_sim = 1.0 if song["artist"] == user_prefs.get("favorite_artist", "") else 0.0
    artist_contrib = WEIGHTS["artist"] * artist_sim
    reasons.append(
        f"artist match ({song['artist']}): contrib={artist_contrib:.4f} (w={WEIGHTS['artist']})"
        if artist_sim else f"artist: no match, contrib=0.0000 (w={WEIGHTS['artist']})"
    )
    total += artist_contrib

    # Title — exact match
    title_sim = 1.0 if song["title"] == user_prefs.get("favorite_title", "") else 0.0
    title_contrib = WEIGHTS["title"] * title_sim
    reasons.append(
        f"title match ({song['title']}): contrib={title_contrib:.4f} (w={WEIGHTS['title']})"
        if title_sim else f"title: no match, contrib=0.0000 (w={WEIGHTS['title']})"
    )
    total += title_contrib

    return round(total, 6), reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
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
    prefs = _normalize_prefs(user_prefs)
    scored = [(song, *score_song(prefs, song)) for song in songs]
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return [(song, score, "\n".join(reasons)) for song, score, reasons in ranked[:k]]


# ---------------------------------------------------------------------------
# Manual test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    songs = load_songs(csv_path)

    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    print("\nTop 5 Recommendations:")
    print("-" * 50)
    for song, score, explanation in recommend_songs(user_prefs, songs, k=5):
        print(f"[{score:.4f}] {song['title']} by {song['artist']} ({song['genre']}, {song['mood']})")
        print(explanation)
        print("-" * 50)