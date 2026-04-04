# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

### Understanding of Real-World Systems
From what I researched, recommendation systems are influenced by how the user interacts with the platform. Likes, dislikes, shares, skips, and replays contribute to what is recommended to them. Behind the scenes, a recommendation system uses scores to determine if a song or video would match a user's preferences. Collaborative filtering (using other users' behavior) looks at other similar users and sees additional songs or videos they have listened/watched and recommend those to a user. However, this could not work well with new users since they don't have as much of a prescence in platform. As for content-based filtering, this is more of looking at a user's profile only and determining what they like (kind of what will be done in this project).

### Features to Use

The features I will focus on will be (order in importance):
  - Energy
  - Acousticness
  - Mood
  - Valence
  - Tempo
  - Danceability
  - Genre
  - Artist
  - Title

Features like artist and title make sense to exclude, given the amount of data we have available, but will include them but have them ranked in the bottom with a low weighting.

### Weighted Scoring Formula
Overall song score = sum(weight * similarity for each feature). Weights sum to 1, based on ranking:
- Energy: 0.25
- Acousticness: 0.20
- Mood: 0.15
- Valence: 0.12
- Tempo: 0.10
- Danceability: 0.08
- Genre: 0.04
- Artist: 0.02
- Title: 0.02

### Similarity Scores Formulas
- **Numerical features (energy, acousticness, valence, tempo, danceability)**: Use `similarity = 1 - |song_value - user_preferred|`. For tempo (in BPM, not 0-1), normalize: `similarity = 1 - |tempo - pref| / (max_tempo - min_tempo)` (e.g., min=60, max=200).

- **Categorical features**:
  - **Genre and Mood**: Use fuzzy similarity matrices (based on relatedness, 0-1 scale).
  - **Artist**: Exact match: `similarity = 1` if same artist, else `0`.
  - **Title**: Exact match: `similarity = 1` if same title, else `0`.


### Similarity Matrix for Categorical Features
Since different moods and genres can be similar in the type of songs, having a discrete 1 or 0 if they equal or not wouldn't make sense, so having fuzzy matching based on relatedness makes more sense. Here they are:

#### Genre Similarity Matrix
(17x17, unique genres: ambient, classical, country, edm, electronic, folk, gospel, hip-hop, indie pop, jazz, lofi, metal, pop, reggae, rock, synthwave, world)

| Genre      | ambient | classical | country | edm | electronic | folk | gospel | hip-hop | indie pop | jazz | lofi | metal | pop | reggae | rock | synthwave | world |
|------------|---------|-----------|---------|-----|------------|------|--------|----------|-----------|------|------|-------|-----|--------|------|----------|-------|
| ambient   | 1.0    | 0.5      | 0.3    | 0.5 | 0.6       | 0.4 | 0.3   | 0.2     | 0.3      | 0.4 | 0.8 | 0.1  | 0.2 | 0.3   | 0.2 | 0.7     | 0.4  |
| classical | 0.5    | 1.0      | 0.4    | 0.2 | 0.3       | 0.6 | 0.7   | 0.1     | 0.5      | 0.8 | 0.4 | 0.2  | 0.4 | 0.3   | 0.3 | 0.2     | 0.5  |
| country   | 0.3    | 0.4      | 1.0    | 0.2 | 0.2       | 0.8 | 0.5   | 0.3     | 0.6      | 0.4 | 0.3 | 0.4  | 0.5 | 0.4   | 0.7 | 0.1     | 0.4  |
| edm       | 0.5    | 0.2      | 0.2    | 1.0 | 0.9       | 0.2 | 0.3   | 0.6     | 0.4      | 0.2 | 0.4 | 0.3  | 0.7 | 0.4   | 0.4 | 0.8     | 0.3  |
| electronic| 0.6    | 0.3      | 0.2    | 0.9 | 1.0       | 0.3 | 0.3   | 0.5     | 0.4      | 0.3 | 0.5 | 0.3  | 0.6 | 0.4   | 0.4 | 0.8     | 0.4  |
| folk      | 0.4    | 0.6      | 0.8    | 0.2 | 0.3       | 1.0 | 0.5   | 0.2     | 0.7      | 0.5 | 0.5 | 0.3  | 0.5 | 0.4   | 0.6 | 0.2     | 0.5  |
| gospel    | 0.3    | 0.7      | 0.5    | 0.3 | 0.3       | 0.5 | 1.0   | 0.4     | 0.5      | 0.8 | 0.3 | 0.2  | 0.6 | 0.5   | 0.4 | 0.2     | 0.7  |
| hip-hop   | 0.2    | 0.1      | 0.3    | 0.6 | 0.5       | 0.2 | 0.4   | 1.0     | 0.5      | 0.3 | 0.3 | 0.4  | 0.8 | 0.7   | 0.5 | 0.4     | 0.5  |
| indie pop | 0.3    | 0.5      | 0.6    | 0.4 | 0.4       | 0.7 | 0.5   | 0.5     | 1.0      | 0.6 | 0.4 | 0.5  | 0.9 | 0.4   | 0.7 | 0.4     | 0.4  |
| jazz      | 0.4    | 0.8      | 0.4    | 0.2 | 0.3       | 0.5 | 0.8   | 0.3     | 0.6      | 1.0 | 0.5 | 0.2  | 0.5 | 0.4   | 0.4 | 0.3     | 0.6  |
| lofi      | 0.8    | 0.4      | 0.3    | 0.4 | 0.5       | 0.5 | 0.3   | 0.3     | 0.4      | 0.5 | 1.0 | 0.2  | 0.3 | 0.3   | 0.3 | 0.5     | 0.4  |
| metal     | 0.1    | 0.2      | 0.4    | 0.3 | 0.3       | 0.3 | 0.2   | 0.4     | 0.5      | 0.2 | 0.2 | 1.0  | 0.4 | 0.2   | 0.9 | 0.3     | 0.3  |
| pop       | 0.2    | 0.4      | 0.5    | 0.7 | 0.6       | 0.5 | 0.6   | 0.8     | 0.9      | 0.5 | 0.3 | 0.4  | 1.0 | 0.5   | 0.6 | 0.6     | 0.4  |
| reggae    | 0.3    | 0.3      | 0.4    | 0.4 | 0.4       | 0.4 | 0.5   | 0.7     | 0.4      | 0.4 | 0.3 | 0.2  | 0.5 | 1.0   | 0.3 | 0.3     | 0.8  |
| rock      | 0.2    | 0.3      | 0.7    | 0.4 | 0.4       | 0.6 | 0.4   | 0.5     | 0.7      | 0.4 | 0.3 | 0.9  | 0.6 | 0.3   | 1.0 | 0.4     | 0.4  |
| synthwave | 0.7    | 0.2      | 0.1    | 0.8 | 0.8       | 0.2 | 0.2   | 0.4     | 0.4      | 0.3 | 0.5 | 0.3  | 0.6 | 0.3   | 0.4 | 1.0     | 0.3  |
| world     | 0.4    | 0.5      | 0.4    | 0.3 | 0.4       | 0.5 | 0.7   | 0.5     | 0.4      | 0.6 | 0.4 | 0.3  | 0.4 | 0.8   | 0.4 | 0.3     | 1.0  |

#### Mood Similarity Matrix
(16x16, unique moods: chill, energetic, exhilarated, focused, happy, hopeful, inspired, intense, laid-back, melancholic, moody, nostalgic, peaceful, playful, relaxed, tribal)

| Mood       | chill | energetic | exhilarated | focused | happy | hopeful | inspired | intense | laid-back | melancholic | moody | nostalgic | peaceful | playful | relaxed | tribal |
|------------|-------|-----------|-------------|---------|-------|---------|----------|---------|-----------|-------------|-------|-----------|----------|---------|---------|--------|
| chill     | 1.0  | 0.4      | 0.3        | 0.8    | 0.5  | 0.6    | 0.5     | 0.3    | 0.9      | 0.6        | 0.7  | 0.7      | 0.9     | 0.4    | 0.9    | 0.4   |
| energetic | 0.4  | 1.0      | 0.8        | 0.6    | 0.7  | 0.7    | 0.6     | 0.9    | 0.5      | 0.3        | 0.4  | 0.4      | 0.4     | 0.8    | 0.5    | 0.6   |
| exhilarated| 0.3  | 0.8      | 1.0        | 0.5    | 0.8  | 0.7    | 0.7     | 0.9    | 0.4      | 0.2        | 0.3  | 0.3      | 0.3     | 0.9    | 0.4    | 0.5   |
| focused   | 0.8  | 0.6      | 0.5        | 1.0    | 0.4  | 0.6    | 0.6     | 0.5    | 0.7      | 0.5        | 0.6  | 0.6      | 0.7     | 0.4    | 0.7    | 0.4   |
| happy     | 0.5  | 0.7      | 0.8        | 0.4    | 1.0  | 0.8    | 0.8     | 0.6    | 0.6      | 0.3        | 0.4  | 0.5      | 0.5     | 0.9    | 0.6    | 0.5   |
| hopeful   | 0.6  | 0.7      | 0.7        | 0.6    | 0.8  | 1.0    | 0.9     | 0.5    | 0.6      | 0.4        | 0.5  | 0.7      | 0.6     | 0.7    | 0.7    | 0.5   |
| inspired  | 0.5  | 0.6      | 0.7        | 0.6    | 0.8  | 0.9    | 1.0     | 0.5    | 0.5      | 0.4        | 0.5  | 0.8      | 0.5     | 0.7    | 0.6    | 0.6   |
| intense   | 0.3  | 0.9      | 0.9        | 0.5    | 0.6  | 0.5    | 0.5     | 1.0    | 0.4      | 0.7        | 0.8  | 0.4      | 0.3     | 0.6    | 0.4    | 0.7   |
| laid-back | 0.9  | 0.5      | 0.4        | 0.7    | 0.6  | 0.6    | 0.5     | 0.4    | 1.0      | 0.5        | 0.6  | 0.6      | 0.8     | 0.5    | 0.9    | 0.5   |
| melancholic| 0.6  | 0.3      | 0.2        | 0.5    | 0.3  | 0.4    | 0.4     | 0.7    | 0.5      | 1.0        | 0.9  | 0.7      | 0.6     | 0.2    | 0.5    | 0.4   |
| moody     | 0.7  | 0.4      | 0.3        | 0.6    | 0.4  | 0.5    | 0.5     | 0.8    | 0.6      | 0.9        | 1.0  | 0.6      | 0.6     | 0.3    | 0.6    | 0.5   |
| nostalgic | 0.7  | 0.4      | 0.3        | 0.6    | 0.5  | 0.7    | 0.8     | 0.4    | 0.6      | 0.7        | 0.6  | 1.0      | 0.7     | 0.4    | 0.7    | 0.5   |
| peaceful  | 0.9  | 0.4      | 0.3        | 0.7    | 0.5  | 0.6    | 0.5     | 0.3    | 0.8      | 0.6        | 0.6  | 0.7      | 1.0     | 0.4    | 0.9    | 0.4   |
| playful   | 0.4  | 0.8      | 0.9        | 0.4    | 0.9  | 0.7    | 0.7     | 0.6    | 0.5      | 0.2        | 0.3  | 0.4      | 0.4     | 1.0    | 0.5    | 0.6   |
| relaxed   | 0.9  | 0.5      | 0.4        | 0.7    | 0.6  | 0.7    | 0.6     | 0.4    | 0.9      | 0.5        | 0.6  | 0.7      | 0.9     | 0.5    | 1.0    | 0.5   |
| tribal    | 0.4  | 0.6      | 0.5        | 0.4    | 0.5  | 0.5    | 0.6     | 0.7    | 0.5      | 0.4        | 0.5  | 0.5      | 0.4     | 0.6    | 0.5    | 1.0   |

### User Profile Information
User profile should have their own scores like preferred_energy, preferred_valence, preferred_acousticness, preferred_tempo, preferred_danceability, preferred_genres, and preferred_moods. These should be scores on the range of [0, 1] (for numerical features) and categorical sets for genre/mood.

Example:

```python
user_profile = {
    'preferred_energy': 0.8,
    'preferred_valence': 0.7,
    'preferred_acousticness': 0.3,
    'preferred_tempo': 120,
    'preferred_danceability': 0.6,
    'preferred_genres': {'pop': 0.5, 'rock': 0.3, 'electronic': 0.2},
    'preferred_moods': {'happy': 0.4, 'energetic': 0.3, 'chill': 0.3},
    'favorite_artist': 'Neon Echo',
    'favorite_genre': 'pop',
    'favorite_mood': 'happy'
}
```

#### Mathematical profile update rule
For numeric features, use an exponential moving average (EMA) update after each user action:

- `pref_new = (1 - \alpha) * pref_old + \alpha * song_value`
- `\alpha` in (0,1] is the learning rate (e.g., 0.1)

Example for energy on a like:

- `pref_energy_new = 0.9 * pref_energy_old + 0.1 * song_energy`

For negative signals (skip/dislike), push away:

- `pref_energy_new = pref_energy_old + \beta * (pref_energy_old - song_energy)`

or just reduce weight by decreasing confidence and normalizing.

For categorical preferences, use counts and soft probabilities:

- `genre_score[genre] += 1` on like, `-= 1` on skip (floor 0)
- normalize to probabilities in [0,1]

#### Ranking rule (song list)
Compute `overall_score` for each candidate song, then sort by descending score:

- `ranked_songs = sorted(songs, key=lambda s: s.overall_score, reverse=True)`

This is the core pipeline: profile update → scoring rule → ranking rule.

### Which songs are chosen? (ranking system)
Based on the similarity scores and weighting system. They should lead to different scores that allows for a ranking system to be used to order songs in how related they are to the user's profile.

Here is an example of it:

![Screenshot](screenshots\recommendations-part-1.png)
![Screenshot](screenshots\recommendations-part-2.png)


### Potential Biases
- Data representation bias: Song catalog may underrepresent diverse genres, artists, or cultures, favoring mainstream options.
Feature weighting bias: Heavier emphasis on features like energy can marginalize calmer or niche songs.

- Similarity matrix bias: Subjective genre/mood relatedness may undervalue emerging or hybrid styles.

- User profile echo chamber: Updates reinforce existing preferences, limiting exposure to new music.

- Cold start bias: New users/songs lack data, leading to suboptimal recommendations.

- Interaction interpretation bias: Skips treated as dislikes may misinterpret user intent.

- Demographic bias: Dataset reflecting specific tastes can exclude or unfairly serve other groups.


---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

Here, I tried multiple experiments such as removing a feature, changing the weight of a feature(s), or trying different edge cases of user profiles. Further discussion is in the model card.

### Feature Removal
Removed danceability (weight=0.08) to see if it had any effect on the recommendations. My prediction was that it wouldn't since it had a lower weight compared to other features. However, there were some changes to "Dead Center" and "Chill Lofi", which points to the idea of danceability being a tiebreaker for some songs. So, lower weighted features are important, by acting as tiebreakers when higher weighted features' scores are tied.

### Weight Change
Changed weights of energy (weight=0.25) and mood (weight=0.15). This made a difference in "Conflicting Energy/Mood" since originally energy was the deciding factor with its higher weight, but with the weight change, mood became the deciding factor.

### User Profiles Tested

#### High Energy
![Screenshot](ai110-module3show-musicrecommendersimulation-starter\screenshots\high_energy_part1.png)
![Screenshot](ai110-module3show-musicrecommendersimulation-starter\screenshots\high_energy_part2.png)

#### Chilled Lofi
![Screenshot](ai110-module3show-musicrecommendersimulation-starter\screenshots\chilled_lofi_part1.png)
![Screenshot](ai110-module3show-musicrecommendersimulation-starter\screenshots\chilled_lofi_part2.png)


#### Deep Intense Rock
![Screenshot](ai110-module3show-musicrecommendersimulation-starter\screenshots\deep_intense_rock_part1.png)
![Screenshot](ai110-module3show-musicrecommendersimulation-starter\screenshots\deep_intense_rock_part2.png)

#### Conflicting Mood/Energy Edge Case
![Screenshot](ai110-module3show-musicrecommendersimulation-starter\screenshots\conflicting_energy_mood_part1.png)
![Screenshot](ai110-module3show-musicrecommendersimulation-starter\screenshots\conflicting_energy_mood_part2.png)

#### Dead Center Edge Case
![Screenshot](ai110-module3show-musicrecommendersimulation-starter\screenshots\dead_center_part1.png)
![Screenshot](ai110-module3show-musicrecommendersimulation-starter\screenshots\dead_center_part2.png)

#### Impossible Unicorn Edge Case
![Screenshot](ai110-module3show-musicrecommendersimulation-starter\screenshots\impossible_unicorn_part1.png)
![Screenshot](ai110-module3show-musicrecommendersimulation-starter\screenshots\impossible_unicorn_part2.png)

## Limitations and Risks

- Similarity Matrix Adaptability
- Dataset Limitation
- Bias towards higher weighted features

(Further discussion in model card)

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

