"""
tests/test_recommender.py
=========================
Tests for the Music Recommender System.

Covers:
1. UserProfile initialization (LLM call)
2. check_ranked_recommendations (LLM call)
3. get_explanations (LLM call)
4. score_song correctness
5. recommend_songs pipeline
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from recommender import UserProfile, Recommender, load_songs, structure_recommendations_for_llm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_PROFILE_DATA = {
    "preferred_energy": 0.7,
    "preferred_acousticness": 0.3,
    "preferred_valence": 0.6,
    "preferred_tempo": 120.0,
    "preferred_danceability": 0.6,
    "preferred_speechiness": 0.1,
    "preferred_loudness": -10.0,
    "preferred_liveness": 0.2
}

MOCK_SONG = {
    "id": "abc123",
    "title": "Test Song",
    "artist": "Test Artist",
    "genre": "Rock",
    "energy": 0.8,
    "acousticness": 0.2,
    "valence": 0.7,
    "tempo": 130.0,
    "danceability": 0.6,
    "loudness": -8.0,
    "liveness": 0.15,
    "speechiness": 0.05,
    **{f"cluster_{i}": 0.1 for i in range(10)}
}

MOCK_RECOMMENDATIONS = [
    (MOCK_SONG, 0.85, "energy: sim=0.900, contrib=0.1980\nacousticness: sim=0.900, contrib=0.1620"),
    ({**MOCK_SONG, "id": "def456", "title": "Song 2"}, 0.75, "energy: sim=0.800, contrib=0.1760"),
    ({**MOCK_SONG, "id": "ghi789", "title": "Song 3"}, 0.65, "energy: sim=0.700, contrib=0.1540"),
    ({**MOCK_SONG, "id": "jkl012", "title": "Song 4"}, 0.55, "energy: sim=0.600, contrib=0.1320"),
    ({**MOCK_SONG, "id": "mno345", "title": "Song 5"}, 0.45, "energy: sim=0.500, contrib=0.1100"),
]


def make_mock_response(content: dict) -> MagicMock:
    """Helper to create a mock Anthropic API response."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = json.dumps(content)
    return mock_response


def make_user_profile() -> UserProfile:
    """Helper to create a UserProfile without hitting the API."""
    with patch("recommender.client.messages.create") as mock_create:
        mock_create.return_value = make_mock_response(MOCK_PROFILE_DATA)
        user = UserProfile(
            favorite_genres=["Rock", "Jazz"],
            favorite_artists=["Test Artist"]
        )
    return user


# ---------------------------------------------------------------------------
# 1. UserProfile initialization LLM call
# ---------------------------------------------------------------------------

def test_user_profile_init_calls_llm():
    """UserProfile.__init__ should call the LLM once and parse the response."""
    with patch("recommender.client.messages.create") as mock_create:
        mock_create.return_value = make_mock_response(MOCK_PROFILE_DATA)
        user = UserProfile(
            favorite_genres=["Rock", "Jazz"],
            favorite_artists=["Test Artist"]
        )
        assert mock_create.call_count == 1
        assert user.preferred_energy == 0.7
        assert user.preferred_acousticness == 0.3
        assert user.preferred_valence == 0.6
        assert user.preferred_tempo == 120.0
        assert user.preferred_danceability == 0.6
        assert user.preferred_speechiness == 0.1
        assert user.preferred_loudness == -10.0
        assert user.preferred_liveness == 0.2


def test_user_profile_init_uses_defaults_on_missing_fields():
    """UserProfile.__init__ should use default values if LLM omits fields."""
    with patch("recommender.client.messages.create") as mock_create:
        mock_create.return_value = make_mock_response({"preferred_energy": 0.8})
        user = UserProfile(favorite_genres=["Pop"], favorite_artists=["Artist"])
        assert user.preferred_energy == 0.8
        assert user.preferred_acousticness == 0.5
        assert user.preferred_tempo == 120.0


# ---------------------------------------------------------------------------
# 2. check_ranked_recommendations LLM calls
# ---------------------------------------------------------------------------

def test_check_ranked_recommendations_high_reliability():
    """check_ranked_recommendations should not call weight adjustment LLM if reliability >= 0.70."""
    user = make_user_profile()
    user.ranked_songs = MOCK_RECOMMENDATIONS

    with patch("recommender.client.messages.create") as mock_create:
        mock_create.return_value = make_mock_response({
            "reliability_score": 0.85,
            "contradictions": []
        })
        user.check_ranked_recommendations(MOCK_RECOMMENDATIONS)
        assert mock_create.call_count == 1


def test_check_ranked_recommendations_low_reliability_triggers_weight_adjustment():
    """check_ranked_recommendations should call weight adjustment LLM if reliability < 0.70."""
    user = make_user_profile()
    user.ranked_songs = MOCK_RECOMMENDATIONS

    mock_weights = {
        "energy": 0.20, "acousticness": 0.18, "valence": 0.13,
        "tempo": 0.10, "danceability": 0.08, "loudness": 0.10,
        "liveness": 0.06, "speechiness": 0.05, "genre": 0.06,
        "artist": 0.02, "title": 0.02
    }

    with patch("recommender.client.messages.create") as mock_create:
        mock_create.side_effect = [
            make_mock_response({"reliability_score": 0.50, "contradictions": ["Energy mismatch"]}),
            make_mock_response(mock_weights)
        ]
        user.check_ranked_recommendations(MOCK_RECOMMENDATIONS)
        assert mock_create.call_count == 2
        assert user.weights == mock_weights


def test_check_ranked_recommendations_weights_sum_to_one():
    """Adjusted weights from LLM should sum to 1.0."""
    user = make_user_profile()

    mock_weights = {
        "energy": 0.20, "acousticness": 0.18, "valence": 0.13,
        "tempo": 0.10, "danceability": 0.08, "loudness": 0.10,
        "liveness": 0.06, "speechiness": 0.05, "genre": 0.06,
        "artist": 0.02, "title": 0.02
    }

    with patch("recommender.client.messages.create") as mock_create:
        mock_create.side_effect = [
            make_mock_response({"reliability_score": 0.50, "contradictions": ["test"]}),
            make_mock_response(mock_weights)
        ]
        user.check_ranked_recommendations(MOCK_RECOMMENDATIONS)
        assert abs(sum(user.weights.values()) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# 3. get_explanations LLM call
# ---------------------------------------------------------------------------

def test_get_explanations_returns_dict():
    """get_explanations should return a dict mapping song IDs to explanations."""
    user = make_user_profile()

    mock_explanations = {
        "abc123": "This song matches your high energy preference.",
        "def456": "Good valence match for your upbeat taste."
    }

    with patch("recommender.client.messages.create") as mock_create:
        mock_create.return_value = make_mock_response(mock_explanations)

        mock_gmm = MagicMock()
        mock_scaler = MagicMock()
        mock_gmm.n_components = 10

        with patch("joblib.load", side_effect=[mock_gmm, mock_scaler]):
            recommender = Recommender([MOCK_SONG])
            result = recommender.get_explanations(user, MOCK_RECOMMENDATIONS[:2])

        assert result == mock_explanations
        assert mock_create.call_count == 1


def test_get_explanations_called_once_per_batch():
    """get_explanations should make one LLM call regardless of number of songs."""
    user = make_user_profile()

    mock_explanations = {song["id"]: "Explanation" for song, _, _ in MOCK_RECOMMENDATIONS}

    with patch("recommender.client.messages.create") as mock_create:
        mock_create.return_value = make_mock_response(mock_explanations)

        mock_gmm = MagicMock()
        mock_scaler = MagicMock()
        mock_gmm.n_components = 10

        with patch("joblib.load", side_effect=[mock_gmm, mock_scaler]):
            recommender = Recommender([MOCK_SONG])
            recommender.get_explanations(user, MOCK_RECOMMENDATIONS)

        assert mock_create.call_count == 1


# ---------------------------------------------------------------------------
# 4. score_song correctness
# ---------------------------------------------------------------------------

def test_score_song_perfect_match():
    """score_song should return a high score when song matches user preferences exactly."""
    user = make_user_profile()
    user.preferred_energy = MOCK_SONG["energy"]
    user.preferred_acousticness = MOCK_SONG["acousticness"]
    user.preferred_valence = MOCK_SONG["valence"]
    user.preferred_tempo = MOCK_SONG["tempo"]
    user.preferred_danceability = MOCK_SONG["danceability"]
    user.preferred_loudness = MOCK_SONG["loudness"]
    user.preferred_liveness = MOCK_SONG["liveness"]
    user.preferred_speechiness = MOCK_SONG["speechiness"]
    user.favorite_genres = [MOCK_SONG["genre"]]
    user.favorite_artists = [MOCK_SONG["artist"]]

    mock_gmm = MagicMock()
    mock_scaler = MagicMock()
    mock_gmm.n_components = 10

    with patch("joblib.load", side_effect=[mock_gmm, mock_scaler]):
        recommender = Recommender([MOCK_SONG])
        score, reasons = recommender.score_song(user, MOCK_SONG)

    assert score > 0.9
    assert len(reasons) > 0


def test_score_song_no_match():
    """score_song should return a low score when song is opposite of user preferences."""
    user = make_user_profile()
    user.preferred_energy = 0.0
    user.preferred_acousticness = 1.0
    user.preferred_valence = 0.0
    user.preferred_tempo = 60.0
    user.preferred_danceability = 0.0
    user.preferred_loudness = -60.0
    user.preferred_liveness = 0.0
    user.preferred_speechiness = 0.0
    user.favorite_genres = ["Classical"]
    user.favorite_artists = []

    opposite_song = {**MOCK_SONG, "energy": 1.0, "acousticness": 0.0, "valence": 1.0,
                     "tempo": 200.0, "danceability": 1.0, "loudness": 0.0,
                     "liveness": 1.0, "speechiness": 1.0, "genre": "Rock"}

    mock_gmm = MagicMock()
    mock_scaler = MagicMock()
    mock_gmm.n_components = 10

    with patch("joblib.load", side_effect=[mock_gmm, mock_scaler]):
        recommender = Recommender([opposite_song])
        score, reasons = recommender.score_song(user, opposite_song)

    assert score < 0.3


def test_score_song_returns_reasons_for_all_features():
    """score_song should return one reason line per feature."""
    user = make_user_profile()

    mock_gmm = MagicMock()
    mock_scaler = MagicMock()
    mock_gmm.n_components = 10

    with patch("joblib.load", side_effect=[mock_gmm, mock_scaler]):
        recommender = Recommender([MOCK_SONG])
        score, reasons = recommender.score_song(user, MOCK_SONG)

    feature_labels = ["energy", "acousticness", "valence", "danceability",
                      "liveness", "speechiness", "loudness", "tempo", "genre", "artist", "title"]
    for label in feature_labels:
        assert any(label in r for r in reasons), f"Missing reason for {label}"


def test_score_song_score_in_valid_range():
    """score_song should always return a score between 0 and 1."""
    user = make_user_profile()

    mock_gmm = MagicMock()
    mock_scaler = MagicMock()
    mock_gmm.n_components = 10

    with patch("joblib.load", side_effect=[mock_gmm, mock_scaler]):
        recommender = Recommender([MOCK_SONG])
        score, _ = recommender.score_song(user, MOCK_SONG)

    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 5. recommend_songs pipeline
# ---------------------------------------------------------------------------

def test_recommend_songs_returns_k_results():
    """recommend_songs should return exactly k results."""
    user = make_user_profile()
    # fix in recommend_songs tests
    songs = [{**MOCK_SONG, "id": str(i)} for i in range(20)]

    mock_gmm = MagicMock()
    mock_scaler = MagicMock()
    mock_gmm.n_components = 10
    mock_gmm.predict_proba.return_value = [[0.1] * 10]
    mock_scaler.transform.return_value = [[0.0] * 8]

    mock_explanations = {song["id"]: "Explanation" for song in songs[:5]}

    with patch("joblib.load", side_effect=[mock_gmm, mock_scaler]):
        recommender = Recommender(songs)
        with patch("recommender.client.messages.create") as mock_create:
            mock_create.return_value = make_mock_response(mock_explanations)
            results = recommender.recommend_songs(user, songs, k=5)

    assert len(results) == 5


def test_recommend_songs_sorted_by_score_descending():
    """recommend_songs should return results sorted by score descending."""
    user = make_user_profile()
    # fix in recommend_songs tests
    songs = [{**MOCK_SONG, "id": str(i)} for i in range(20)]    

    mock_gmm = MagicMock()
    mock_scaler = MagicMock()
    mock_gmm.n_components = 10
    mock_gmm.predict_proba.return_value = [[0.1] * 10]
    mock_scaler.transform.return_value = [[0.0] * 8]

    mock_explanations = {song["id"]: "Explanation" for song in songs[:5]}

    with patch("joblib.load", side_effect=[mock_gmm, mock_scaler]):
        recommender = Recommender(songs)
        with patch("recommender.client.messages.create") as mock_create:
            mock_create.return_value = make_mock_response(mock_explanations)
            results = recommender.recommend_songs(user, songs, k=5)

    scores = [score for _, score, _ in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 6. EMA updates (like and skip)
# ---------------------------------------------------------------------------

def test_like_nudges_preferences_toward_song():
    """like() should nudge all preferences toward the liked song's values."""
    
    user = make_user_profile()
    user.preferred_energy = 0.5
    user.preferred_acousticness = 0.5
    user.preferred_valence = 0.5
    user.preferred_tempo = 120.0
    user.preferred_danceability = 0.5
    user.preferred_loudness = -20.0
    user.preferred_liveness = 0.5
    user.preferred_speechiness = 0.5

    high_energy_song = {**MOCK_SONG, "energy": 1.0, "acousticness": 1.0,
                        "valence": 1.0, "tempo": 200.0, "danceability": 1.0,
                        "loudness": 0.0, "liveness": 1.0, "speechiness": 1.0}

    # fix all like/skip tests — wrap the call like this
    with patch("recommender.client.messages.create") as mock_create:
        mock_create.return_value = make_mock_response({"reliability_score": 0.9, "contradictions": []})
        user.like(high_energy_song)  # or user.skip(), user.like(MOCK_SONG), etc.

    assert user.preferred_energy > 0.5
    assert user.preferred_acousticness > 0.5
    assert user.preferred_valence > 0.5
    assert user.preferred_tempo > 120.0
    assert user.preferred_danceability > 0.5
    assert user.preferred_loudness > -20.0
    assert user.preferred_liveness > 0.5
    assert user.preferred_speechiness > 0.5


def test_skip_pushes_preferences_away_from_song():
    """skip() should push all preferences away from the skipped song's values."""

    user = make_user_profile()
    user.preferred_energy = 0.5
    user.preferred_acousticness = 0.5
    user.preferred_valence = 0.5
    user.preferred_tempo = 120.0
    user.preferred_danceability = 0.5
    user.preferred_loudness = -20.0
    user.preferred_liveness = 0.5
    user.preferred_speechiness = 0.5

    high_energy_song = {**MOCK_SONG, "energy": 1.0, "acousticness": 1.0,
                        "valence": 1.0, "tempo": 200.0, "danceability": 1.0,
                        "loudness": 0.0, "liveness": 1.0, "speechiness": 1.0}

    # fix all like/skip tests — wrap the call like this
    with patch("recommender.client.messages.create") as mock_create:
        mock_create.return_value = make_mock_response({"reliability_score": 0.9, "contradictions": []})
        user.skip(high_energy_song)  # or user.skip(), user.like(MOCK_SONG), etc.

    assert user.preferred_energy < 0.5
    assert user.preferred_acousticness < 0.5
    assert user.preferred_valence < 0.5
    assert user.preferred_tempo < 120.0
    assert user.preferred_danceability < 0.5
    assert user.preferred_loudness < -20.0
    assert user.preferred_liveness < 0.5
    assert user.preferred_speechiness < 0.5


def test_like_uses_alpha_correctly():
    """like() EMA formula: new = alpha * song + (1 - alpha) * old."""
    user = make_user_profile()
    user.preferred_energy = 0.5
    user.alpha = 0.1

    song = {**MOCK_SONG, "energy": 1.0}
    # fix all like/skip tests — wrap the call like this
    with patch("recommender.client.messages.create") as mock_create:
        mock_create.return_value = make_mock_response({"reliability_score": 0.9, "contradictions": []})
        user.like(song)  # or user.skip(), user.like(MOCK_SONG), etc.

    expected = 0.1 * 1.0 + 0.9 * 0.5
    assert abs(user.preferred_energy - expected) < 0.001


def test_skip_uses_beta_correctly():
    """skip() push formula: new = (1 + beta) * old - beta * song."""
    user = make_user_profile()
    user.preferred_energy = 0.5
    user.beta = 0.1

    song = {**MOCK_SONG, "energy": 1.0}
    # fix all like/skip tests — wrap the call like this
    with patch("recommender.client.messages.create") as mock_create:
        mock_create.return_value = make_mock_response({"reliability_score": 0.9, "contradictions": []})
        user.skip(song)  # or user.skip(), user.like(MOCK_SONG), etc.

    expected = 1.1 * 0.5 - 0.1 * 1.0
    assert abs(user.preferred_energy - expected) < 0.001


def test_like_increments_interaction_count():
    """like() should increment num_interactions by 1."""
    user = make_user_profile()
    initial = user.num_interactions
    # fix all like/skip tests — wrap the call like this
    with patch("recommender.client.messages.create") as mock_create:
        mock_create.return_value = make_mock_response({"reliability_score": 0.9, "contradictions": []})
        user.like(MOCK_SONG)  # or user.skip(), user.like(MOCK_SONG), etc.
    assert user.num_interactions == initial + 1


def test_skip_increments_interaction_count():
    """skip() should increment num_interactions by 1."""
    user = make_user_profile()
    initial = user.num_interactions
    # fix all like/skip tests — wrap the call like this
    with patch("recommender.client.messages.create") as mock_create:
        mock_create.return_value = make_mock_response({"reliability_score": 0.9, "contradictions": []})
        user.skip(MOCK_SONG)  # or user.skip(), user.like(MOCK_SONG), etc.
    assert user.num_interactions == initial + 1


def test_like_appends_to_log():
    """like() should add an entry to the log."""
    user = make_user_profile()
    # fix all like/skip tests — wrap the call like this
    with patch("recommender.client.messages.create") as mock_create:
        mock_create.return_value = make_mock_response({"reliability_score": 0.9, "contradictions": []})
        user.like(MOCK_SONG)  # or user.skip(), user.like(MOCK_SONG), etc.
    assert any("Liked" in entry for entry in user.log)


def test_skip_appends_to_log():
    """skip() should add an entry to the log."""
    user = make_user_profile()
    with patch("recommender.client.messages.create") as mock_create:
        mock_create.return_value = make_mock_response({"reliability_score": 0.9, "contradictions": []})
        user.skip(MOCK_SONG)  # or user.skip(), user.like(MOCK_SONG), etc.
    assert any("Skipped" in entry for entry in user.log)


# ---------------------------------------------------------------------------
# 7. Weight validation
# ---------------------------------------------------------------------------

def test_default_weights_sum_to_one():
    """Default weights in UserProfile should sum to 1.0."""
    user = make_user_profile()
    assert abs(sum(user.weights.values()) - 1.0) < 0.001


def test_update_weights_replaces_weights():
    """update_weights() should fully replace the weights dict."""
    user = make_user_profile()
    new_weights = {
        "energy": 0.20, "acousticness": 0.18, "valence": 0.13,
        "tempo": 0.10, "danceability": 0.08, "loudness": 0.10,
        "liveness": 0.06, "speechiness": 0.05, "genre": 0.06,
        "artist": 0.02, "title": 0.02
    }
    user.update_weights(new_weights)
    assert user.weights == new_weights


def test_updated_weights_sum_to_one():
    """Manually updated weights should still sum to 1.0."""
    user = make_user_profile()
    new_weights = {
        "energy": 0.20, "acousticness": 0.18, "valence": 0.13,
        "tempo": 0.10, "danceability": 0.08, "loudness": 0.10,
        "liveness": 0.06, "speechiness": 0.05, "genre": 0.06,
        "artist": 0.02, "title": 0.02
    }
    user.update_weights(new_weights)
    assert abs(sum(user.weights.values()) - 1.0) < 0.001

