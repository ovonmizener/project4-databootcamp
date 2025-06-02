from flask import Flask, render_template, request
import os
import re
import pickle
from textblob import TextBlob
import sqlite3
import datetime
from feature_predictor import FeaturePredictor

# test

# Initialize the Flask app.
app = Flask(__name__)

def load_pickled_models():
    """
    Load the TF-IDF vectorizers and predictive models from the 'resources' folder.
    Returns:
        tfidf_sentiment, sentiment_model, tfidf_genre, genre_model, feature_predictor
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
            
        # Load feature predictor
        feature_predictor = FeaturePredictor()
        feature_predictor.load_models()
        
        return tfidf_sentiment, sentiment_model, tfidf_genre, genre_model, feature_predictor
    except FileNotFoundError as e:
        print("Error loading pickle file:", e)
        return None, None, None, None, None

# Load the models.
tfidf_sentiment, sentiment_model, tfidf_genre, genre_model, feature_predictor = load_pickled_models()
if None in [tfidf_sentiment, sentiment_model, tfidf_genre, genre_model, feature_predictor]:
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
    Predicts sentiment, genre, and additional features for the input lyrics.
    Returns:
        sentiment_prediction: Binary prediction (1 = Positive/Neutral, 0 = Negative).
        sentiment_proba: Model confidence (if available).
        tb_score: TextBlob sentiment polarity.
        genre_prediction: Predicted genre.
        feature_predictions: Dictionary of predicted features (danceability, loudness, etc.)
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
    
    # Predict additional features
    feature_predictions = feature_predictor.predict_features(lyrics)
    
    return sentiment_prediction, sentiment_proba, tb_score, genre_prediction, feature_predictions

def init_db():
    """
    Initializes the submissions database.
    Drops and recreates the table to ensure correct schema.
    """
    db_path = os.path.join("resources", "submissions_log.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Drop the existing table if it exists
    c.execute("DROP TABLE IF EXISTS submissions")
    
    # Create the table with the correct schema
    c.execute("""
        CREATE TABLE submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            artist TEXT,
            lyrics TEXT,
            sentiment TEXT,
            sentiment_confidence REAL,
            textblob_score REAL,
            predicted_genre TEXT,
            danceability REAL,
            loudness REAL,
            acousticness REAL,
            instrumentalness REAL,
            valence REAL,
            energy REAL
        )
    """)
    conn.commit()
    conn.close()

def log_submission_db(artist, lyrics, sentiment, sentiment_confidence, tb_score, genre, features):
    """
    Logs the submission data into the SQLite database.
    """
    db_path = os.path.join("resources", "submissions_log.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    c.execute("""
        INSERT INTO submissions (
            timestamp, artist, lyrics, sentiment, sentiment_confidence, 
            textblob_score, predicted_genre, danceability, loudness, 
            acousticness, instrumentalness, valence, energy
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp, artist, lyrics, sentiment, sentiment_confidence, 
        tb_score, genre, features['danceability'], features['loudness'],
        features['acousticness'], features['instrumentalness'], 
        features['valence'], features['energy']
    ))
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
    
    sentiment_pred, sentiment_conf, tb_score, genre_pred, feature_preds = predict_values(lyrics)
    sentiment_label = "Positive/Neutral" if sentiment_pred == 1 else "Negative"
    
    # Log the submission.
    log_submission_db(artist, lyrics, sentiment_label, sentiment_conf, tb_score, genre_pred, feature_preds)
    
    result = {
        "artist": artist,
        "sentiment": sentiment_label,
        "sentiment_confidence": sentiment_conf,
        "tb_score": tb_score,
        "predicted_genre": genre_pred,
        "lyrics": lyrics,
        "features": feature_preds
    }
    
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
