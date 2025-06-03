# 🎵 Mood-Based Music Recommendation System

This is a local Python application that predicts the **mood of a given text** (like a lyric or sentence) and recommends matching songs using a trained machine learning model and a dataset of Spotify tracks.

You can use this either via the **command line** or a **Streamlit web app** running locally in your browser.

---

## 📁 Project Structure

```
mood-music-recommender/
├── app.py                    # CLI entry point
├── streamlit_app.py          # Streamlit frontend
├── text_mood_predictor.py    # Mood prediction logic
├── spotify_recommender.py    # Song recommendation logic
├── models/
│   └── mood_classifier.pkl   # Pre-trained model (joblib)
├── data/
│   └── spotify_data.csv      # Spotify songs with mood labels
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🛠 Requirements

- Python 3.7+
- `pip` installed
- Internet browser (for Streamlit interface)

---

## 🧪 Installation & Setup

1. **Clone the repo:**

```bash
git clone https://github.com/yourusername/mood-music-recommender.git
cd mood-music-recommender
```

2. **Create a virtual environment (recommended):**

- **macOS/Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- **Windows**:
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 📦 Example `requirements.txt`

Ensure this file includes:

```
scikit-learn
pandas
joblib
streamlit
```

---

## ▶️ Run Locally

### Option 1: Command-Line Interface

```bash
python app.py
```

Sample interaction:

```
🎤 Enter a sentence, lyric, or mood description:
> I feel calm and peaceful.

🎵 Predicted Mood: calm

🎧 Recommended Songs:
1. "Ocean Eyes" by Billie Eilish
2. "River Flows In You" by Yiruma
3. "Let Her Go" by Passenger
```

---

### Option 2: Streamlit Web App

Launch the Streamlit frontend with:

```bash
streamlit run streamlit_app.py
```

It should open in your browser at:

```
http://localhost:8501
```

There, you can:

- Type a sentence describing your mood
- Instantly see the predicted mood
- View a list of mood-matching songs

---

## 🧠 How It Works

- `text_mood_predictor.py`: loads and applies a pre-trained model (`mood_classifier.pkl`) to classify mood from text
- `spotify_recommender.py`: filters Spotify songs by predicted mood
- `app.py`: simple CLI-based interface
- `streamlit_app.py`: user-friendly browser interface via Streamlit

---

## 🌱 Future Improvements

- Use a transformer model like BERT for mood prediction
- Add real-time Spotify API integration
- Include audio-based mood detection from uploaded music

---

## 🙌 Acknowledgments

- scikit-learn for ML modeling
- Streamlit for web-based UI
- Inspired by emotion-aware recommender systems



