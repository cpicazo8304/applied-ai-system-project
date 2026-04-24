from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import pandas as pd
import os


if __name__ == "__main__":
    # Download latest version
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    df = pd.read_csv(dataset_path)

    # Only keep genre, artist, song name, track id, and audio features (acousticness, danceability, energy, valence, tempo)
    features = ["id", "name", "artists", "danceability", "energy", "loudness","speechiness", "acousticness", "liveness", "valence", "tempo", "genre"]
    df = df[features]

    # Convert to numpy array for clustering
    X = df[["danceability", "energy", "loudness", "speechiness", "acousticness", "liveness", "valence", "tempo"]].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, "models/scaler.joblib")  # save scaler too, needed later for user profile

    # soft clustering using Gaussian Mixture Models because it allows for probabilistic cluster assignments, 
    # which can capture the uncertainty in the data and provide more nuanced insights into the relationships 
    # between songs. This is particularly useful in music data, where songs can have overlapping characteristics 
    # and may not fit neatly into distinct clusters. By using GMM, we can better understand the underlying structure 
    # of the data and identify patterns that may not be apparent with hard clustering methods like K-means.
    bic_scores = []
    probabilities = []
    gms = []
    for n in [20, 30, 40, 50]:
        gmm = GaussianMixture(n_components=n)
        gmm.fit(X)
        gms.append(gmm)
        probabilities.append(gmm.predict_proba(X))  # shape (n_samples, n_components)
        bic_scores.append(gmm.bic(X))

    # plot bic_scores
    optimal_idx = np.argmin(bic_scores)
    optimal_probabilities = probabilities[optimal_idx]
    optimal_gmm = gms[optimal_idx]

    # save model for later use
    model_path = os.path.join(os.path.dirname(__file__), "models", "gmm_model.joblib")
    joblib.dump(optimal_gmm, model_path)

    # Save the probabilities into features plus probabilities into a new CSV file
    prob_df = pd.DataFrame(optimal_probabilities, columns=[f"cluster_{i}" for i in range(optimal_probabilities.shape[1])])
    result_df = pd.concat([df, prob_df], axis=1)
    result_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs_with_clusters.csv")
    result_df.to_csv(result_path, index=False)