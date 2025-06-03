import joblib

# Load the model
model = joblib.load("models/mood_classifier.pkl")

def predict_mood(text):
    """
    Predict mood from input text using the trained Logistic Regression model.
    """
    prediction = model.predict([text])[0]
    return prediction

# Example usage
if __name__ == "__main__":
    user_input = input("🎤 Enter a sentence, lyric, or mood description: ")
    mood = predict_mood(user_input)
    print(f"\n🎵 Predicted Mood: {mood}")
