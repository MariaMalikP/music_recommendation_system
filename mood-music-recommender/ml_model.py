from datasets import load_dataset

# Load the GoEmotions dataset (only train set is enough for now)
dataset = load_dataset("go_emotions", split="train")

# Mapping GoEmotions labels to custom mood categories
goemotions_to_mood = {
    'admiration': 'happy',
    'amusement': 'happy',
    'approval': 'happy',
    'caring': 'relaxed',
    'desire': 'happy',
    'excitement': 'energetic',
    'gratitude': 'happy',
    'joy': 'happy',
    'love': 'happy',
    'optimism': 'energetic',
    'pride': 'energetic',
    'relief': 'relaxed',
    'realization': 'energetic',
    'remorse': 'sad',
    'sadness': 'sad',
    'grief': 'sad',
    'disappointment': 'sad',
    'anger': 'sad',
    'disapproval': 'sad',
    'embarrassment': 'sad',
    'nervousness': 'sad',
    'confusion': 'sad',
    'fear': 'sad',
    'annoyance': 'sad',
    'neutral': None  # We'll drop these
}

import pandas as pd

# Load label names
label_names = dataset.features["labels"].feature.names

# Convert to DataFrame
rows = []

for example in dataset:
    text = example['text']
    labels = example['labels']
    moods = set()
    
    for label in labels:
        emotion = label_names[label]
        mood = goemotions_to_mood.get(emotion)
        if mood:
            moods.add(mood)

    # Skip if no mood found or multiple moods
    if len(moods) == 1:
        rows.append({"text": text, "mood": moods.pop()})

df = pd.DataFrame(rows)
print(df.sample(5))

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

df['clean_text'] = df['text'].apply(clean_text)
df = df[['clean_text', 'mood']]
df.to_csv("data/cleaned_mood_dataset.csv", index=False)
print("✅ Saved cleaned dataset to data/cleaned_mood_dataset.csv")
