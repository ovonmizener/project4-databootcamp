import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# ---------- STEP 1: Connect to the Database and Read Data ----------
# Specify the path to your SQLite database file
db_path = "resources/submissions_log.db"

# Establish a connection to the database
conn = sqlite3.connect(db_path)

# Read the entire 'submissions' table into a DataFrame
df = pd.read_sql_query("SELECT * FROM submissions", conn)

# Always close the connection after reading the data
conn.close()

# ---------- STEP 2: Preprocess the Data ----------
# Convert the timestamp column to datetime objects
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create a new 'year' column based on the timestamp
df['year'] = df['timestamp'].dt.year

# Map the sentiment labels to numerical scores:
# For example, "Positive/Neutral" becomes 1 and "Negative" becomes 0.
# This allows us to compute average sentiment per time period.
sentiment_map = {"Positive/Neutral": 1, "Negative": 0}
df['sentiment_numeric'] = df['sentiment'].map(sentiment_map)

# ---------- STEP 3: Filter Data for a Specific Artist ----------
# Replace "Artist Name" with the actual artist you want to analyze.
artist_name = "Artist Name"  # e.g., "Adele"
artist_df = df[df['artist'].str.lower() == artist_name.lower()]

if artist_df.empty:
    print(f"No data found for artist '{artist_name}'. Please choose a different artist.")
    exit()

# ---------- STEP 4: Group By Year to Get Trend Data ----------
# Compute the average sentiment score per year
# (The value will be 1 if all submissions are positive, 0 if all are negative,
#  or somewhere in between.)
trend = artist_df.groupby("year")["sentiment_numeric"].mean().reset_index()

# ---------- STEP 5: Plot the Trend ----------
plt.figure(figsize=(10, 6))
plt.plot(trend["year"], trend["sentiment_numeric"], marker="o", linestyle="-", color="blue")
plt.title(f"Sentiment Trend for {artist_name} Over Time")
plt.xlabel("Year")
plt.ylabel("Average Sentiment (1 = Positive/Neutral, 0 = Negative)")
plt.ylim(0, 1)
plt.grid(True)
plt.show()
