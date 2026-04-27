import anthropic
from dotenv import load_dotenv
import os
import json
from typing import List
from recommender import Recommender, load_songs, UserProfile, structure_recommendations_for_llm

def user_profile_test(client):
    """Test that the user profile can be created and updated correctly."""
    favorite_artists = ["Taylor Swift", "The Weeknd", "Dua Lipa"]
    favorite_genres = ["Pop", "Rock", "Jazz"]

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

    raw = response.content[0].text
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    profile_data = json.loads(clean)
    print("Generated User Profile:\n", json.dumps(profile_data, indent=2))
    preferred_energy = profile_data.get("preferred_energy", 0.5)
    preferred_acousticness = profile_data.get("preferred_acousticness", 0.5)
    preferred_valence = profile_data.get("preferred_valence", 0.5)
    preferred_tempo = profile_data.get("preferred_tempo", 120.0)
    preferred_danceability = profile_data.get("preferred_danceability", 0.5)
    preferred_speechiness = profile_data.get("preferred_speechiness", 0.5)
    preferred_loudness = profile_data.get("preferred_loudness", -10.0)
    preferred_liveness = profile_data.get("preferred_liveness", 0.5)

    assert 0 <= preferred_energy <= 1, "Preferred energy must be between 0 and 1"
    assert 0 <= preferred_acousticness <= 1, "Preferred acousticness must be between 0 and 1"
    assert 0 <= preferred_valence <= 1, "Preferred valence must be between 0 and 1"
    assert preferred_tempo > 0, "Preferred tempo must be a positive number"
    assert 0 <= preferred_danceability <= 1, "Preferred danceability must be between 0 and 1"
    assert 0 <= preferred_speechiness <= 1, "Preferred speechiness must be between 0 and 1"
    assert preferred_loudness < 0, "Preferred loudness must be a negative number"
    assert 0 <= preferred_liveness <= 1, "Preferred liveness must be between 0 and 1"


def test_structure_recommendations_for_LLM():
    # Test the structure_recommendations_for_llm function

    # recommendations : List of (song_dict, score, explanation) tuples.
    # clusters add to 1
    recommendations = [
        (
            {"title": "Song A", "artist": "Artist A", "genre": "Pop", "id": "123", "danceability": 0.8, "energy": 0.9, "acousticness": 0.1, "valence": 0.7, "tempo": 130.0, "speechiness": 0.05, "loudness": -5.0, "liveness": 0.2, "cluster_1": 0.5, "cluster_2": 0.3, "cluster_3": 0.2 },
            0.95,
            "This song matches your preference for high energy and pop genre."
        ),
        (
            {"title": "Song B", "artist": "Artist B", "genre": "Rock", "id": "456", "danceability": 0.6, "energy": 0.8, "acousticness": 0.3, "valence": 0.5, "tempo": 120.0, "speechiness": 0.1, "loudness": -15.0, "liveness": 0.4, "cluster_1": 0.4, "cluster_2": 0.4, "cluster_3": 0.2 },
            0.90,
            "This song matches your preference for high energy and rock genre."
        ),
        (
            {"title": "Song C", "artist": "Artist C", "genre": "Jazz", "id": "789", "danceability": 0.4, "energy": 0.6, "acousticness": 0.7, "valence": 0.3, "tempo": 110.0, "speechiness": 0.2, "loudness": -20.0, "liveness": 0.6, "cluster_1": 0.3, "cluster_2": 0.5, "cluster_3": 0.2 },
            0.85,
            "This song matches your preference for acousticness and jazz genre."
        )
    ]
    structured = structure_recommendations_for_llm(recommendations)
    print("Structured Recommendations:\n", structured)

    
def test_get_explanations(client):
    recommendations = [
        (
            {"title": "Song A", "artist": "Artist A", "genre": "Pop", "id": "123", "danceability": 0.8, "energy": 0.9, "acousticness": 0.1, "valence": 0.7, "tempo": 130.0, "speechiness": 0.05, "loudness": -5.0, "liveness": 0.2, "cluster_1": 0.5, "cluster_2": 0.3, "cluster_3": 0.2 },
            0.95,
            "This song matches your preference for high energy and pop genre."
        ),
        (
            {"title": "Song B", "artist": "Artist B", "genre": "Rock", "id": "456", "danceability": 0.6, "energy": 0.8, "acousticness": 0.3, "valence": 0.5, "tempo": 120.0, "speechiness": 0.1, "loudness": -15.0, "liveness": 0.4, "cluster_1": 0.4, "cluster_2": 0.4, "cluster_3": 0.2 },
            0.90,
            "This song matches your preference for high energy and rock genre."
        ),
        (
            {"title": "Song C", "artist": "Artist C", "genre": "Jazz", "id": "789", "danceability": 0.4, "energy": 0.6, "acousticness": 0.7, "valence": 0.3, "tempo": 110.0, "speechiness": 0.2, "loudness": -20.0, "liveness": 0.6, "cluster_1": 0.3, "cluster_2": 0.5, "cluster_3": 0.2 },
            0.85,
            "This song matches your preference for acousticness and jazz genre."
        )
    ]
    user_prefs = UserProfile(
        favorite_artists = ["Taylor Swift", "The Weeknd", "Dua Lipa"],
        favorite_genres = ["Pop", "Rock", "Jazz"]
    )
    recommendations_structured = structure_recommendations_for_llm(recommendations)
    prompt = f"""Given the following ranked recommendations with their feature contributions, generate a plain English explanation for why each song was ranked where it was. 
    Focus on the most influential features and how they align or misalign with the user's preferences. Make the explanations user-friendly and concise, avoiding technical jargon.
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
    raw = response.content[0].text
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    explanations = json.loads(clean)
    
    for song_id, explanation in explanations.items():
        print(f"Song ID: {song_id}\nExplanation: {explanation}\n")


def test_check_ranked_recommendations(client):
    recommendations = [
        (
            {"title": "Song A", "artist": "Artist A", "genre": "Pop", "id": "123", "danceability": 0.8, "energy": 0.9, "acousticness": 0.1, "valence": 0.7, "tempo": 130.0, "speechiness": 0.05, "loudness": -5.0, "liveness": 0.2, "cluster_1": 0.5, "cluster_2": 0.3, "cluster_3": 0.2 },
            0.95,
            "This song matches your preference for high energy and pop genre."
        ),
        (
            {"title": "Song B", "artist": "Artist B", "genre": "Rock", "id": "456", "danceability": 0.6, "energy": 0.8, "acousticness": 0.3, "valence": 0.5, "tempo": 120.0, "speechiness": 0.1, "loudness": -15.0, "liveness": 0.4, "cluster_1": 0.4, "cluster_2": 0.4, "cluster_3": 0.2 },
            0.90,
            "This song matches your preference for high energy and rock genre."
        ),
        (
            {"title": "Song C", "artist": "Artist C", "genre": "Jazz", "id": "789", "danceability": 0.4, "energy": 0.6, "acousticness": 0.7, "valence": 0.3, "tempo": 110.0, "speechiness": 0.2, "loudness": -20.0, "liveness": 0.6, "cluster_1": 0.3, "cluster_2": 0.5, "cluster_3": 0.2 },
            0.85,
            "This song matches your preference for acousticness and jazz genre."
        )
    ]
    recommendations_structured = structure_recommendations_for_llm(recommendations)
    user_prefs = UserProfile(
        favorite_artists = ["Taylor Swift", "The Weeknd", "Dua Lipa"],
        favorite_genres = ["Pop", "Rock", "Jazz"]
    )
    user_prefs_structured = user_prefs.structure_profile()
    prompt1 = f""""
    Given top 5 recommendations with the scores, and user profile, check if the rankings make 
    logical sense given the user's preferences. Flag any contradictions and produce a reliability 
    score (0-1) for the overall recommendation set such as a high energy song ranking highly for a 
    low energy user because tempo dominated.
    Here are the recommendations with scores: {recommendations_structured}. 
    Here is the user profile: {user_prefs_structured}.
    Respond in JSON format with no preamble with keys "reliability_score" (float between 0 and 1) and 
    "contradictions" (list of strings describing any contradictions found, empty if none found).
    """

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt1}
        ]
    )
    raw = response.content[0].text
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    result = json.loads(clean)
    reliability_score = result.get("reliability_score", 0.0)
    contradictions = result.get("contradictions", [])

    print(f"Reliability Score: {reliability_score}")
    if contradictions:
        print("Contradictions found:")
        for contradiction in contradictions:
            print(f"- {contradiction}")
    else:
        print("No contradictions found.")


def test_changing_weights(client):
    user_profile = UserProfile(
        favorite_artists = ["Taylor Swift", "The Weeknd", "Dua Lipa"],
        favorite_genres = ["Pop", "Rock", "Jazz"]
    )
    user_prefs_structured = user_profile.structure_profile()
    reliability_score = 0.4
    contradictions = [
        "Song A is ranked highest but has very high energy which contradicts the user's low preferred energy.",
        "Song C is ranked lowest but has high acousticness which contradicts the user's preference for acoustic songs."
    ]
    prompt2 = f"""Given a not good enough reliability score {reliability_score}, can you change the 
    given weights to better suit the current user? Here are the contradictions: {contradictions}. 
    Here are the current weights: {user_profile.get_weights()}. Here is the user profile: {user_prefs_structured}. 
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
    raw = response.content[0].text
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    new_weights = json.loads(clean)

    print("Old Weights:\n", json.dumps(user_profile.get_weights(), indent=2))
    print("New Weights:\n", json.dumps(new_weights, indent=2))

    # Update weights with new values
    user_profile.update_weights(new_weights)

    total = sum(user_profile.get_weights().values())
    if abs(total - 1.0) > 0.01:
        print(f"Warning: adjusted weights sum to {total:.3f}, not 1.0")



if __name__ == "__main__":
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "env", ".env"))
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Run the user profile test
    print("\nRunning user_profile_test...\n")
    user_profile_test(client)

    # run the get explanations test
    print()
    print("\nRunning test_get_explanations...\n")
    test_get_explanations(client)

    # run the check ranked recommendations test
    print()
    print("\nRunning test_check_ranked_recommendations...\n")
    test_check_ranked_recommendations(client)

    # run the changing weights test
    print()
    print("\nRunning test_changing_weights...\n")
    test_changing_weights(client)

