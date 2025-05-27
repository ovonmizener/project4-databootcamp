🎶 Music Sentiment Analyzer 🎶  

Group Members 
Oliver Von Mizener
Justin Wright
Lynn Nguyen


<b><big>Project Overview</big></b>
  Our group aims to create a web tool that can predict lyric sentiment based off a Spotify dataset we have sourced. We’ll be leveraging machine learning to create a pipeline that takes the raw data, cleans it, performs a sentiment analysis which gives us a base set of data to train from. The predictive model will then be trained on assessing sentiment scores based off of text given, we will use Flask for a front end application. Users will be able to input lyrics, as well as have access to a visual dashboard to understand what this score means.


Data Sources 
  XXXXX 

Tech Stack 
- NoSQL
- Flask 

Process
  Data Collection:
    Gather a Spotify dataset containing song lyrics.
  Data Cleaning:
    Use Python libraries (like Pandas and NLTK) to preprocess and clean the text.
  Baseline Sentiment:
    Apply TextBlob to generate initial sentiment scores for training labels.
  Model Training:
    Convert text to numerical features with TfidfVectorizer and train a sentiment classifier (e.g., Logistic Regression) using scikit‑learn.
  Deployment:
    Serialize models with pickle and build a Flask web app that:
    Accepts user-input lyrics.
    Displays the predicted sentiment and related visualizations.
