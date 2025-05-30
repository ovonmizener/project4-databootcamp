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
  [model files].pkl     (Pre-trained .pkl files for predictive models - required)
  dashboardapp.py          (Dash dashboard application)
  app.py             (Flask application for user submissions, if separate)
  requirements.txt   (Dependencies to install)
  README.txt               (This file)

NOTE: The pre-trained model files (.pkl) are required for the tool to run correctly. They should be included in the repository or instructions provided on how to obtain them.

HOW TO RUN
----------
1. Clone the Repository:
   git clone https://github.com/yourusername/Music-Sentiment-Analyzer.git
   cd Music-Sentiment-Analyzer

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
     * submissions_log.db (for user submissions)
     * All required .pkl model files

4. Run the Dashboard:
   python dashboardapp.py
   (The dashboard should be accessible at http://127.0.0.1:8050/)

5. (Optional) Run the Flask Application:
   If you want to run the front end submission, in a new terminal, run:
   python flask_app.py

PROCESS & WORKFLOW
------------------
1. Data Collection & Cleaning:
   - Gather the Spotify dataset and preprocess it using Python.
   - Convert the release_date column (which contains only the year) to integers.
   - Remove unwanted characters and standardize the text in the lyrics.
2. Baseline Sentiment Analysis:
   - Use TextBlob to generate initial sentiment scores as training labels.
   - Train a sentiment classifier (e.g., Logistic Regression using scikit‑learn) and serialize it with pickle.
3. Visualization & Dashboard Deployment:
   - Pre-compute static figures (e.g., average sentiment by genre; scatter plot for release year vs. sentiment).
   - Build an interactive dashboard with multiple tabs (Overview, Numeric Analysis, Thematic Analysis, Track Info, User Responses).
   - Integrate a Flask user submission interface to capture additional lyric data dynamically.
4. User Interaction:
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