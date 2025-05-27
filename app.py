from flask import Flask, render_template, request
import os
import re
import pickle
from textblob import TextBlob

# Initialize the Flask app.
app = Flask(__name__)

# ---------------------------
# Pickle Explanation:
# ---------------------------
# The pickle module is used to serialize and deserialize Python objects.
# Here, we're using it to load our pre-trained TF-IDF vectorizer and sentiment model
# from disk so we don't have to rebuild or retrain the model each time the app starts.
# ---------------------------

# Load the TF-IDF vectorizer and sentiment model from the 'resources' folder.
# The 'resources' folder should contain 'tfidf.pkl' and 'sentiment_model.pkl'.
def load_pickled_model():
    """
    Load the TF-IDF vectorizer and sentiment model from the 'resources' folder.
    Returns:
        tfidf: The loaded TfidfVectorizer.
        sentiment_model: The loaded sentiment prediction model.
    """
    vectorizer_path = os.path.join("resources", "tfidf.pkl")
    model_path = os.path.join("resources", "sentiment_model.pkl")
    try:
        with open(vectorizer_path, "rb") as f:
            tfidf = pickle.load(f)
        with open(model_path, "rb") as f:
            sentiment_model = pickle.load(f)
        return tfidf, sentiment_model
    except FileNotFoundError as e:
        print("Error loading pickle file:", e)
        return None, None

# Load the pickled objects.
tfidf, sentiment_model = load_pickled_model()
if tfidf is None or sentiment_model is None:
    raise Exception("Pickle files not found. Please run your training script to generate 'tfidf.pkl' and 'sentiment_model.pkl' in the 'resources' folder.")

def clean_text(text):
    """
    Simple text cleaning function:
      - Converts text to lowercase.
      - Removes punctuation.
      - Splits the text into tokens and then rejoins them.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return " ".join(tokens)

def predict_sentiment(lyrics):
    """
    Predicts the sentiment of provided lyrics.
    Uses the loaded TF-IDF vectorizer and sentiment model.
    Also calculates a TextBlob sentiment score for additional context.
    Returns:
        prediction: The predicted sentiment label.
        proba: The confidence/probability of the prediction (if available).
        tb_score: The TextBlob sentiment polarity (for interpretability).
    """
    cleaned = clean_text(lyrics)
    features = tfidf.transform([cleaned])
    prediction = sentiment_model.predict(features)[0]
    try:
        # If the model supports predict_proba, get the probability of the predicted class.
        proba = sentiment_model.predict_proba(features)[0, prediction]
    except AttributeError:
        proba = None
    tb_score = TextBlob(cleaned).sentiment.polarity
    return prediction, proba, tb_score

@app.route("/")
def index():
    # Render the home page.
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the lyrics entered by the user from the form.
    lyrics = request.form.get("lyrics")
    if not lyrics or lyrics.strip() == "":
        error_msg = "Please enter some lyrics to get a prediction."
        return render_template("index.html", error=error_msg)
    
    # Use our prediction function.
    prediction, proba, tb_score = predict_sentiment(lyrics)
    sentiment = "Positive/Neutral" if prediction == 1 else "Negative"
    result = {
        "sentiment": sentiment,
        "confidence": proba,
        "tb_score": tb_score,
        "lyrics": lyrics
    }
    # Display the result to the user.
    return render_template("result.html", result=result)

if __name__ == "__main__":
    # Run the Flask app.
    app.run(debug=True)
