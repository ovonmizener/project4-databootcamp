import os
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px


# Ensure your processed CSV file is in the resources folder.
data_file_path = os.path.join("resources", "processed_tcc_ceds_music.csv")
df = pd.read_csv(data_file_path)

# If release_date is available, extract release_year
if "release_date" in df.columns:
    df["release_year"] = pd.to_datetime(df["release_date"], errors='coerce').dt.year


# Chart 1: Average sentiment score by genre
if "genre" in df.columns and "sentiment_score" in df.columns:
    genre_sentiment_df = df.groupby("genre")["sentiment_score"].mean().reset_index()
    fig_genre_sentiment = px.bar(
        genre_sentiment_df,
        x="genre",
        y="sentiment_score",
        title="Average Sentiment Score by Genre",
        labels={"sentiment_score": "Average Sentiment Score", "genre": "Genre"}
    )
else:
    fig_genre_sentiment = {}

# Chart 2: Numeric Feature Distribution
# Define a list of numeric features that you want to explore.
numeric_cols = ["danceability", "loudness", "acousticness", "instrumentalness", "valence", "energy"]
# Set an initial feature for the histogram.
initial_feature = numeric_cols[0] if numeric_cols else None

# Chart 3: Release Year vs. Sentiment Score Scatterplot
if df["release_year"].notna().sum() > 0 and "sentiment_score" in df.columns:
    fig_release_sentiment = px.scatter(
        df,
        x="release_year",
        y="sentiment_score",
        hover_data=["artist_name", "track_name"],
        title="Release Year vs. Sentiment Score",
        labels={"release_year": "Release Year", "sentiment_score": "Sentiment Score"}
    )
else:
    fig_release_sentiment = {}

# Dash App Layout
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Music Dataset Trends Dashboard", style={"textAlign": "center"}),

    # Average sentiment by genre section
    html.Div([
        html.H2("Average Sentiment Score by Genre"),
        dcc.Graph(
            id="genre-sentiment-graph",
            figure=fig_genre_sentiment
        )
    ], style={"width": "80%", "margin": "auto"}),

    # Numeric feature distribution section
    html.Div([
        html.H2("Numeric Feature Distribution"),
        dcc.Dropdown(
            id="numeric-feature-dropdown",
            options=[{"label": col.capitalize(), "value": col} for col in numeric_cols],
            value=initial_feature,
            clearable=False,
            style={"width": "50%", "margin": "auto"}
        ),
        dcc.Graph(
            id="numeric-feature-histogram"
        )
    ], style={"width": "80%", "margin": "auto"}),

    # Release year vs. sentiment scatter section
    html.Div([
        html.H2("Release Year vs. Sentiment Score"),
        dcc.Graph(
            id="release-sentiment-scatter",
            figure=fig_release_sentiment
        )
    ], style={"width": "80%", "margin": "auto", "paddingBottom": "2em"})
])


# Dash Callbacks
@app.callback(
    Output("numeric-feature-histogram", "figure"),
    [Input("numeric-feature-dropdown", "value")]
)
def update_histogram(selected_feature):
    """
    Update the histogram based on the selected numeric feature.
    """
    if selected_feature:
        fig = px.histogram(df, x=selected_feature, nbins=30,
                           title=f"Distribution of {selected_feature.capitalize()}",
                           labels={selected_feature: selected_feature.capitalize()})
        return fig
    return {}

if __name__ == '__main__':
    # Updated to use app.run() according to latest Dash versions
    app.run(debug=True)
