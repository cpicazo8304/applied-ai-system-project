# 🎧 Model Card: AuraTrack Applied AI Music Recommender

## 1. Model Name

**AuraTrack 2.0**

## 2. Intended Use

This recommender is designed to suggest songs from a 550k+ song catalog that best match a user's listening preferences based on audio features and genre. It assumes the user has some general sense of what they like in terms of genres and favorite artists. This is primarily a classroom exploration project built to demonstrate how a production-style recommender pipeline works in practice, combining machine learning clustering, weighted similarity scoring, and LLM-powered self-critique. The architecture mirrors real-world music recommenders used by platforms like Spotify.

## 3. How the Model Works

AuraTrack uses a two-stage retrieval and ranking pipeline:

**Stage 1 — Candidate Generation:** A Gaussian Mixture Model (GMM) trained on 550k+ songs clusters songs using Expectation Maximization — a soft clustering approach where each song has a probability distribution across all clusters rather than a hard assignment. When a user profile is created, their preferences are converted into a cluster probability vector and compared against all songs using cosine similarity. The top 100 candidates are retrieved for detailed scoring.

**Stage 2 — Weighted Similarity Scoring:** Each candidate is scored across 11 features: energy, acousticness, valence, tempo, danceability, loudness, liveness, speechiness, genre, artist, and title. Numerical features use direct subtraction, tempo and loudness are normalized over their ranges, and genre uses a fuzzy similarity matrix. Each user has their own personalized weight set that adapts over time.

**LLM Self-Critique:** After ranking, Claude evaluates whether the recommendations make logical sense given the user profile and returns a reliability score. If the score falls below 0.70, a second LLM call suggests personalized weight adjustments — meaning the scoring formula adapts per user rather than using global fixed weights.

**Agentic Loop:** As the user likes and skips songs, their profile updates via Exponential Moving Average (EMA), nudging preferences toward liked songs and away from skipped ones. Significant shifts trigger a reliability check to keep recommendations aligned with evolving tastes.

**Profile Initialization:** When a user provides their favorite genres and artists, an LLM infers their numerical preferences (energy, acousticness, valence, tempo, danceability, loudness, liveness, speechiness) rather than requiring manual input.

## 4. Data

- The catalog contains 550k+ songs collected from a Spotify Kaggle dataset with features: id, title, artist, genre, energy, tempo, valence, danceability, acousticness, loudness, liveness, and speechiness.
- Genres represented (10 total): Rock, Hip-Hop, Classical, Pop, Jazz, R&B, Blues, Electronic, Country, and Folk.
- A GMM was trained offline on the full catalog using the 8 numerical audio features and saved via joblib. Cluster probability vectors for each song are stored in the CSV so no training happens at runtime.
- Missing aspects:
    - Genre coverage: only 10 genres compared to the diversity of real music catalogs.
    - No mood labels: mood was removed due to data availability constraints, reducing the expressiveness of the similarity scoring.
    - Sub-genre nuance: no distinction between hard rock and indie rock, or deep house and tech EDM.
    - Lyrics: could add additional signal for mood and theme matching.

## 5. Strengths

The two-stage GMM + weighted scoring pipeline scales to 550k+ songs efficiently — the GMM narrows candidates in milliseconds using vectorized cosine similarity, and detailed scoring only runs on 100 candidates rather than the full catalog. The LLM self-critique layer catches cases where the math produces misleading results, such as a high energy song ranking highly for a low energy user because one feature dominated. The adaptive weight system means each user's scoring formula personalizes over time rather than applying the same global weights
