import pandas as pd

def load_dataset(path="data/dataset.csv"):
    df = pd.read_csv(path)

    df = df.drop_duplicates(subset=['track_name', 'artists'])

    features = ['track_name', 'artists', 'valence', 'energy', 'danceability', 'tempo']
    df = df[features]

    return df
import random

# Define mood-to-feature mappings
mood_feature_filters = {
    "happy": {"valence_min": 0.7, "energy_min": 0.6},
    "sad": {"valence_max": 0.4, "energy_max": 0.5},
    "energetic": {"energy_min": 0.8},
    "relaxed": {"energy_max": 0.5, "valence_min": 0.5},
    "angry": {"energy_min": 0.7, "valence_max": 0.4}
}

def recommend_songs(df, mood, n=10):
    filters = mood_feature_filters.get(mood)
    if not filters:
        print(f"No rules defined for mood: {mood}")
        return pd.DataFrame()

    # Apply filters
    recs = df.copy()
    if "valence_min" in filters:
        recs = recs[recs["valence"] >= filters["valence_min"]]
    if "valence_max" in filters:
        recs = recs[recs["valence"] <= filters["valence_max"]]
    if "energy_min" in filters:
        recs = recs[recs["energy"] >= filters["energy_min"]]
    if "energy_max" in filters:
        recs = recs[recs["energy"] <= filters["energy_max"]]

    if recs.empty:
        print("😕 No matching songs found.")
        return pd.DataFrame()

    return recs.sample(n=min(n, len(recs)))
import joblib

model = joblib.load("models/mood_classifier.pkl")

def predict_mood(text):
    """
    Predict mood from input text using the trained model.
    """
    return model.predict([text])[0]
