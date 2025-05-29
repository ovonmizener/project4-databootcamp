from flask import Flask, render_template, request
import os
import re
import pickle
from textblob import TextBlob
import sqlite3
import datetime

# test

# Initialize the Flask app.
app = Flask(__name__)

def load_pickled_models():
    """
    Load the TF-IDF vectorizers and predictive models from the 'resources' folder.
    Returns:
        tfidf_sentiment, sentiment_model, tfidf_genre, genre_model
    """
    resources_folder = "resources"
    try:
        with open(os.path.join(resources_folder, "tfidf_sentiment.pkl"), "rb") as f:
            tfidf_sentiment = pickle.load(f)
        with open(os.path.join(resources_folder, "sentiment_model.pkl"), "rb") as f:
            sentiment_model = pickle.load(f)
        with open(os.path.join(resources_folder, "tfidf_genre.pkl"), "rb") as f:
            tfidf_genre = pickle.load(f)
        with open(os.path.join(resources_folder, "genre_model.pkl"), "rb") as f:
            genre_model = pickle.load(f)
        return tfidf_sentiment, sentiment_model, tfidf_genre, genre_model
    except FileNotFoundError as e:
        print("Error loading pickle file:", e)
        return None, None, None, None

# Load the models.
tfidf_sentiment, sentiment_model, tfidf_genre, genre_model = load_pickled_models()
if None in [tfidf_sentiment, sentiment_model, tfidf_genre, genre_model]:
    raise Exception("One or more pickle files not found. Ensure the training script has been run successfully.")

def clean_text(text):
    """
    Clean incoming text: convert to lowercase, remove punctuation, and tokenize.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

def predict_values(lyrics):
    """
    Predicts sentiment and genre for the input lyrics.
    Returns:
        sentiment_prediction: Binary prediction (1 = Positive/Neutral, 0 = Negative).
        sentiment_proba: Model confidence (if available).
        tb_score: TextBlob sentiment polarity.
        genre_prediction: Predicted genre.
    """
    cleaned = clean_text(lyrics)
    
    # Predict sentiment.
    features_sent = tfidf_sentiment.transform([cleaned])
    sentiment_prediction = sentiment_model.predict(features_sent)[0]
    try:
        sentiment_proba = sentiment_model.predict_proba(features_sent)[0, sentiment_prediction]
    except AttributeError:
        sentiment_proba = None
      
    tb_score = TextBlob(cleaned).sentiment.polarity
    
    # Predict genre.
    features_genre = tfidf_genre.transform([cleaned])
    genre_prediction = genre_model.predict(features_genre)[0]
    
    return sentiment_prediction, sentiment_proba, tb_score, genre_prediction

def init_db():
    """
    Initializes the submissions database.
    Uses CREATE TABLE IF NOT EXISTS to avoid dropping existing data.
    """
    db_path = os.path.join("resources", "submissions_log.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Create the table only if it doesn't exist.
    c.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            artist TEXT,
            lyrics TEXT,
            sentiment TEXT,
            sentiment_confidence REAL,
            textblob_score REAL,
            predicted_genre TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_submission_db(artist, lyrics, sentiment, sentiment_confidence, tb_score, genre):
    """
    Logs the submission data into the SQLite database.
    """
    db_path = os.path.join("resources", "submissions_log.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    c.execute("""
        INSERT INTO submissions (timestamp, artist, lyrics, sentiment, sentiment_confidence, textblob_score, predicted_genre)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, artist, lyrics, sentiment, sentiment_confidence, tb_score, genre))
    conn.commit()
    conn.close()

# Initialize the database (this will NOT clear existing data)
init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    artist = request.form.get("artist")
    lyrics = request.form.get("lyrics")
    if not lyrics or lyrics.strip() == "":
        error_msg = "Please enter some lyrics to get a prediction."
        return render_template("index.html", error=error_msg)
    
    sentiment_pred, sentiment_conf, tb_score, genre_pred = predict_values(lyrics)
    sentiment_label = "Positive/Neutral" if sentiment_pred == 1 else "Negative"
    
    # Log the submission.
    log_submission_db(artist, lyrics, sentiment_label, sentiment_conf, tb_score, genre_pred)
    
    result = {
        "artist": artist,
        "sentiment": sentiment_label,
        "sentiment_confidence": sentiment_conf,
        "tb_score": tb_score,
        "predicted_genre": genre_pred,
        "lyrics": lyrics
    }
    
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
