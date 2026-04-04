"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main(user_profile=None) -> None:
    songs = load_songs("../data/songs.csv") 

    # Starter example profile
    user_prefs = user_profile if user_profile is not None else {"genre": "pop", "mood": "happy", "energy": 0.8}
    

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()


if __name__ == "__main__":
    # 1. High-Energy Pop
    high_energy_pop = {
        "preferred_energy": 0.9,
        "preferred_acousticness": 0.1,
        "preferred_valence": 0.85,
        "preferred_tempo": 128.0,
        "preferred_danceability": 0.85,
        "preferred_genres": {"pop": 0.6, "edm": 0.3, "indie pop": 0.1},
        "preferred_moods": {"energetic": 0.5, "happy": 0.3, "exhilarated": 0.2},
        "favorite_artist": "",
        "favorite_title": "",
    }

    # 2. Chill Lofi
    chill_lofi = {
        "preferred_energy": 0.35,
        "preferred_acousticness": 0.80,
        "preferred_valence": 0.55,
        "preferred_tempo": 75.0,
        "preferred_danceability": 0.50,
        "preferred_genres": {"lofi": 0.6, "ambient": 0.3, "jazz": 0.1},
        "preferred_moods": {"chill": 0.5, "focused": 0.3, "peaceful": 0.2},
        "favorite_artist": "",
        "favorite_title": "",
    }

    # 3. Deep Intense Rock
    deep_intense_rock = {
        "preferred_energy": 0.92,
        "preferred_acousticness": 0.08,
        "preferred_valence": 0.35,
        "preferred_tempo": 145.0,
        "preferred_danceability": 0.55,
        "preferred_genres": {"rock": 0.5, "metal": 0.4, "electronic": 0.1},
        "preferred_moods": {"intense": 0.5, "moody": 0.3, "melancholic": 0.2},
        "favorite_artist": "",
        "favorite_title": "",
    }

    # 4. Conflicting energy vs mood — high energy but wants melancholic/peaceful mood.
    #    Scores will be pulled in opposite directions. Watch which weight wins.
    conflicting_energy_mood = {
        "preferred_energy": 0.95,
        "preferred_acousticness": 0.2,
        "preferred_valence": 0.2,
        "preferred_tempo": 140.0,
        "preferred_danceability": 0.5,
        "preferred_genres": {"rock": 0.5, "ambient": 0.5},  # also contradictory
        "preferred_moods": {"melancholic": 0.6, "peaceful": 0.4},
        "favorite_artist": "",
        "favorite_title": "",
    }

    # 5. All preferences at 0.5 (dead center) — every song will score similarly.
    #    Tests whether your ranking produces a meaningful spread or a flat tie.
    dead_center = {
        "preferred_energy": 0.5,
        "preferred_acousticness": 0.5,
        "preferred_valence": 0.5,
        "preferred_tempo": 130.0,
        "preferred_danceability": 0.5,
        "preferred_genres": {"pop": 0.25, "rock": 0.25, "lofi": 0.25, "jazz": 0.25},
        "preferred_moods": {"chill": 0.25, "happy": 0.25, "intense": 0.25, "relaxed": 0.25},
        "favorite_artist": "",
        "favorite_title": "",
    }

    # 6. Impossible unicorn — preferences that no single song in the CSV satisfies.
    #    High energy AND high acousticness AND slow tempo. Should still rank something
    #    on top, but scores will be universally low.
    impossible_unicorn = {
        "preferred_energy": 0.95,
        "preferred_acousticness": 0.95,
        "preferred_valence": 0.95,
        "preferred_tempo": 65.0,
        "preferred_danceability": 0.95,
        "preferred_genres": {"classical": 1.0},
        "preferred_moods": {"exhilarated": 1.0},
        "favorite_artist": "",
        "favorite_title": "",
    }
    
    profiles = {
        "High-Energy Pop": high_energy_pop,
        "Chill Lofi": chill_lofi,
        "Deep Intense Rock": deep_intense_rock,
        "Conflicting Energy/Mood": conflicting_energy_mood,
        "Dead Center": dead_center,
        "Impossible Unicorn": impossible_unicorn,
    }

    for name, profile in profiles.items():
        print(f"\n{'='*50}")
        print(f"Profile: {name}")
        print('='*50)
        main(user_profile=profile)