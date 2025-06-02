import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
from textblob import TextBlob
import re
from nltk.corpus import stopwords
import nltk
import sys
import time

print("Starting feature predictor training...")

# Download required NLTK data
try:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    print("NLTK stopwords downloaded successfully.")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    sys.exit(1)

class FeaturePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.tfidf = None
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return " ".join(filtered_tokens)
    
    def prepare_features(self, df):
        print("Preparing features...")
        # Clean lyrics
        print("Cleaning lyrics...")
        df['clean_lyrics'] = df['lyrics'].apply(self.clean_text)
        
        # Extract text features
        print("Extracting text features...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        X_text = self.tfidf.fit_transform(df['clean_lyrics'])
        
        # Convert to dense array for combining with other features
        X_text_dense = X_text.toarray()
        
        # Get sentiment scores
        print("Calculating sentiment scores...")
        df['sentiment'] = df['clean_lyrics'].apply(lambda x: TextBlob(x).sentiment.polarity)
        
        # Combine text features with sentiment
        X = np.hstack([X_text_dense, df['sentiment'].values.reshape(-1, 1)])
        print("Feature preparation complete.")
        return X
    
    def train_models(self, data_path):
        print(f"\nLoading data from {data_path}...")
        try:
            # Load data
            df = pd.read_csv(data_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Features to predict
        target_features = [
            'danceability', 'loudness', 'acousticness',
            'instrumentalness', 'valence', 'energy'
        ]
        
        print("\nTraining models for each feature...")
        # Train a model for each feature
        for feature in target_features:
            print(f"\nTraining model for {feature}...")
            if feature not in df.columns:
                print(f"Warning: {feature} not found in dataset")
                continue
                
            y = df[feature]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            print(f"Scaling features for {feature}...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model with progress reporting
            print(f"Training Random Forest model for {feature}...")
            start_time = time.time()
            
            # Use fewer trees and enable parallel processing
            model = RandomForestRegressor(
                n_estimators=50,  # Reduced from 100 to 50
                random_state=42,
                n_jobs=-1,  # Use all available CPU cores
                verbose=1  # Enable built-in progress reporting
            )
            
            model.fit(X_train_scaled, y_train)
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Evaluate
            print(f"Evaluating {feature} model...")
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\nResults for {feature}:")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R2 Score: {r2:.4f}")
            
            # Save model and scaler
            self.models[feature] = model
            self.scalers[feature] = scaler
            print(f"Model for {feature} saved.")
    
    def predict_features(self, lyrics):
        # Clean and prepare input
        cleaned_lyrics = self.clean_text(lyrics)
        sentiment = TextBlob(cleaned_lyrics).sentiment.polarity
        
        # Transform text
        X_text = self.tfidf.transform([cleaned_lyrics])
        X_text_dense = X_text.toarray()
        
        # Combine with sentiment
        X = np.hstack([X_text_dense, np.array([[sentiment]])])
        
        # Make predictions
        predictions = {}
        for feature, model in self.models.items():
            X_scaled = self.scalers[feature].transform(X)
            pred = model.predict(X_scaled)[0]
            predictions[feature] = pred
            
        return predictions
    
    def save_models(self, directory="resources"):
        print("\nSaving models...")
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save TF-IDF vectorizer
            print("Saving TF-IDF vectorizer...")
            with open(os.path.join(directory, "feature_tfidf.pkl"), "wb") as f:
                pickle.dump(self.tfidf, f)
            
            # Save models and scalers
            for feature in self.models:
                print(f"Saving model and scaler for {feature}...")
                with open(os.path.join(directory, f"feature_model_{feature}.pkl"), "wb") as f:
                    pickle.dump(self.models[feature], f)
                with open(os.path.join(directory, f"feature_scaler_{feature}.pkl"), "wb") as f:
                    pickle.dump(self.scalers[feature], f)
            print("All models saved successfully.")
        except Exception as e:
            print(f"Error saving models: {e}")
            sys.exit(1)
    
    def load_models(self, directory="resources"):
        print("\nLoading models...")
        try:
            # Load TF-IDF vectorizer
            print("Loading TF-IDF vectorizer...")
            with open(os.path.join(directory, "feature_tfidf.pkl"), "rb") as f:
                self.tfidf = pickle.load(f)
            
            # Load models and scalers
            features = ['danceability', 'loudness', 'acousticness', 
                       'instrumentalness', 'valence', 'energy']
            
            for feature in features:
                try:
                    print(f"Loading model and scaler for {feature}...")
                    with open(os.path.join(directory, f"feature_model_{feature}.pkl"), "rb") as f:
                        self.models[feature] = pickle.load(f)
                    with open(os.path.join(directory, f"feature_scaler_{feature}.pkl"), "rb") as f:
                        self.scalers[feature] = pickle.load(f)
                except FileNotFoundError:
                    print(f"Warning: Model for {feature} not found")
            print("All models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)

if __name__ == "__main__":
    try:
        print("Initializing FeaturePredictor...")
        predictor = FeaturePredictor()
        
        data_path = os.path.join("resources", "tcc_ceds_music.csv")
        print(f"Training models using data from: {data_path}")
        predictor.train_models(data_path)
        
        print("\nSaving trained models...")
        predictor.save_models()
        
        print("\nFeature predictor training completed successfully!")
    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1) 