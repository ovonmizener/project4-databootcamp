🎶 Sonic Sentiments - A Music Sentiment Analyzer 🎶  

========================

Group Members:
- Oliver Von Mizener
- Justin Wright
- Lynn Nguyen

PROJECT OVERVIEW
----------------
The Music Sentiment Analyzer is a web-based tool designed to analyze and visualize musical trends through sentiment and genre analysis. The project leverages a curated music dataset (up to 2019) as its data source. We preprocess the raw data, perform sentiment and genre analysis using machine learning techniques, and provide interactive visualizations using a dashboard. In addition, the tool includes a front-end application where users can submit lyrics to receive real-time sentiment feedback.

FEATURES
--------
- Data Processing Pipeline:
  - Extracts key information including the release year directly from the dataset.
  - Cleans and standardizes raw text data.
- Sentiment & Genre Analysis:
  - Uses pre-computed sentiment scores and machine learning models on the lyrics.
  - Visualizes average sentiment scores by genre, correlations among numeric features, and more.
- Interactive Dashboard:
  - Built using Dash and Plotly with multiple tabs: Overview, Numeric Analysis, Thematic Analysis, Track Info, and User Responses.
  - Integrates real-time user submissions from a separate Flask application.
- User Submission Interface:
  - A Flask-based front end allows users to input lyrics.
  - Inputs are processed, stored in an SQLite database, and reflected on the dashboard, providing dynamic insights.

DATA SOURCES
------------
- Primary Dataset:
  - Spotify dataset sourced from Mendeley Data (https://data.mendeley.com/datasets/3t9vbwxgr5/2) containing song metadata and lyrics (up to 2019).
- Supplementary Data:
  - User-submitted lyrics are stored in a local SQLite database ("submissions_log.db").

TECH STACK
----------
- Programming Language: Python 3.8+
- Libraries & Frameworks:
  - Data Processing: Pandas, NumPy
  - Visualization: Plotly, Dash
  - Machine Learning & NLP: scikit-learn, TextBlob
  - Web Application: Flask, Dash
  - Database: SQLite
- Styles & Assets: Custom CSS located in the "assets" folder

PROJECT STRUCTURE
-----------------
project4-databootcamp/
  assets/
    custom.css           (Custom CSS file for the dark gray & blue theme)
  resources/
    processed_tcc_ceds_music.csv   (The base CSV dataset)
    submissions_log.db   (SQLite database for user submissions)
    # Required Model Files:
    feature_tfidf.pkl    (TF-IDF vectorizer for feature prediction)
    feature_model_danceability.pkl
    feature_model_loudness.pkl
    feature_model_acousticness.pkl
    feature_model_instrumentalness.pkl
    feature_model_valence.pkl
    feature_model_energy.pkl
    feature_scaler_danceability.pkl
    feature_scaler_loudness.pkl
    feature_scaler_acousticness.pkl
    feature_scaler_instrumentalness.pkl
    feature_scaler_valence.pkl
    feature_scaler_energy.pkl
    sentiment_model.pkl  (Sentiment analysis model)
    sentiment_scaler.pkl
    sentiment_tfidf.pkl
    genre_model.pkl     (Genre prediction model)
    genre_scaler.pkl
    genre_tfidf.pkl
  dashboardapp.py       (Dash dashboard application)
  app.py               (Flask application for user submissions)
  feature_predictor.py (Script for training feature prediction models)
  sentimentanalyzer.py (Script for training sentiment analysis model)
  genre_predictor.py   (Script for training genre prediction model)
  requirements.txt     (Dependencies to install)
  README.md           (This file)

NOTE: The pre-trained model files (.pkl) are required for the tool to run correctly. They should be included in the repository or instructions provided on how to obtain them.

HOW TO RUN
----------
1. Clone the Repository:
   git clone https://github.com/ovonmizener/project4-databootcamp.git
   cd project4-databootcamp

2. Set Up the Environment:
   - Create and activate a virtual environment (using `venv` or `conda`).
     For example, using venv:
       python -m venv venv
       On Linux/Mac: source venv/bin/activate
       On Windows: venv\\Scripts\\activate
   - Install dependencies:
       pip install -r requirements.txt

3. Prepare the Data Files:
   - Ensure the "resources" folder contains:
     * processed_tcc_ceds_music.csv (the base dataset)
     * All required .pkl model files (see Project Structure section)

4. Run Order (if training models from scratch):
   a. First, train the sentiment analysis model:
      python sentimentanalyzer.py
   b. Then, train the genre prediction model:
      python genre_predictor.py*
   c. Finally, train the feature prediction model:
      python feature_predictor.py
   
   Note: If using pre-trained models, you can skip steps 4a-4c.
   
   *Important: The feature_predictor.py file is included for review purposes only. 
   DO NOT run this file on local machines as it requires significant computational resources 
   and the pre-trained models are already included in the resources folder. This file is 
   provided to demonstrate how the feature prediction models were created and trained. This will take HOURS to run on a standard machine. 

5. Run the Applications:
   a. Start the Flask application (for user submissions):
      python app.py
      (The web interface will be available at http://127.0.0.1:5000/)
   
   b. In a separate terminal, start the dashboard:
      python dashboardapp.py
      (The dashboard will be available at http://127.0.0.1:8050/)

Note: The Flask application (app.py) and the dashboard (dashboardapp.py) can be run in any order, but both need to be running for full functionality.

PROCESS & WORKFLOW
------------------
1. Data Collection & Cleaning:
   - Gather the Spotify dataset and preprocess it using Python.
   - Convert the release_date column (which contains only the year) to integers.
   - Remove unwanted characters and standardize the text in the lyrics.

2. Feature Analysis & Prediction:
   - Implemented a proof-of-concept feature predictor that attempts to predict musical features from lyrics
   - Uses TF-IDF vectorization and Random Forest Regression to predict:
     * Danceability
     * Loudness
     * Acousticness
     * Instrumentalness
     * Valence
     * Energy
   - Note: This is a proof-of-concept implementation with limited accuracy (R² scores around 0.15)
   - Future iterations could improve accuracy by:
     * Using more sophisticated models
     * Incorporating additional features
     * Training on larger datasets

3. Baseline Sentiment Analysis (Primary Graded Component):
   - Implemented a robust sentiment analysis model achieving approximately 87% accuracy
   - Uses TextBlob for initial sentiment scoring and Logistic Regression for classification
   - This is the primary model chosen for grading, demonstrating strong performance in
     predicting sentiment from lyrics
   - The model is trained on a large dataset of song lyrics and their associated
     sentiment scores, making it particularly effective for musical content
   - Results are visualized in the dashboard through various interactive charts
     and user submission analysis

4. Visualization & Dashboard Deployment:
   - Pre-compute static figures (e.g., average sentiment by genre; scatter plot for release year vs. sentiment).
   - Build an interactive dashboard with multiple tabs (Overview, Numeric Analysis, Thematic Analysis, Track Info, User Responses).
   - Integrate a Flask user submission interface to capture additional lyric data dynamically.

5. User Interaction:
   - Allow users to input lyrics through a web form.
   - Process input through pre-trained models and update the dashboard with new insights.

FUTURE ENHANCEMENTS
-------------------
- Integrate professional APIs and additional datasets to expand the analysis scope.
- Explore advanced deep learning techniques for nuanced sentiment and genre classification.
- Improve dashboard interactivity and scalability for real-world applications.

LIMITATIONS
-------
The primary limitation of this project is the dataset. Our analysis is based on a Spotify dataset that only extends to 2019 and includes a limited number of genres and entries, which constrains the overall representativeness of the data. As a result, the sentiment and genre analyses sometimes yield inaccurate or unexpected results (for instance, misclassifications like midwest emo being read as reggae) due to insufficient comparative data. These constraints were accepted to rapidly develop a proof-of-concept, and future iterations will aim to integrate more comprehensive, professional data sources to overcome these shortcomings.

ACKNOWLEDGEMENTS
----------------
Special thanks to our group members, advisors, and the open-source community for their support. Thank you to Paul Arias and Brian Perry, our instructors in the Data Analytics Boot Camp for a great class and all your support. 