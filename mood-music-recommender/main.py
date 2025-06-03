from spotify_recommender import load_dataset, recommend_songs
from text_mood_predictor import predict_mood
def main():
    print("🎤 Enter a sentence, lyric, or mood description:")
    user_input = input("> ")
    
    mood = predict_mood(user_input)
    print(f"\n🎵 Predicted Mood: {mood}\n")

    df = load_dataset()
    recommendations = recommend_songs(df, mood)

    if not recommendations.empty:
        print(f"🎧 Top Songs for Mood '{mood}':\n")
        for i, row in recommendations.iterrows():
            print(f"{row['track_name']} by {row['artists']} 🎶")

if __name__ == "__main__":
    main()
