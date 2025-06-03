import streamlit as st
from text_mood_predictor import predict_mood
from spotify_recommender import load_dataset, recommend_songs

# Streamlit page config
st.set_page_config(page_title="Mood-Based Music Recommender", page_icon="🎵")

# App title
st.title("🎧 Mood-Based Music Recommender")
st.markdown("Type in a lyric, phrase, or description of how you're feeling, and get song recommendations that match your mood!")

# Text input
user_input = st.text_input("🎤 What's on your mind?")
if user_input:
    # Predict mood
    mood = predict_mood(user_input)
    st.success(f"🎵 Predicted Mood: **{mood}**")

    # Load songs dataset and get recommendations
    df = load_dataset()
    recommendations = recommend_songs(df, mood)

    # Show recommendations
    if not recommendations.empty:
        st.subheader(f"🎶 Top Song Recommendations for '{mood}' Mood:")
        for _, row in recommendations.iterrows():
            st.markdown(f"**{row['track_name']}** by *{row['artists']}*")
    else:
        st.warning("No songs found for this mood. Try a different input.")
