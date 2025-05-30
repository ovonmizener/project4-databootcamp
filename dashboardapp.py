import os
import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import sqlite3

# -----------------------------
# Load and Process Dataset
# -----------------------------
data_file_path = os.path.join("resources", "processed_tcc_ceds_music.csv")
df = pd.read_csv(data_file_path)

# Convert release_date (which is just the year, e.g., "1950") to integer.
if "release_date" in df.columns:
    df["release_year"] = df["release_date"].astype(int)

# -----------------------------
# Pre-compute Static Figures for the Overview Tab
# -----------------------------
# (1) Average Sentiment Score by Genre
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

# (2) Release Year vs. Sentiment Score Scatterplot
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

# (3) Correlation Heatmap of Numeric Features
corr_df = df.select_dtypes(include=["number"]).corr()
fig_corr = px.imshow(
    corr_df,
    text_auto=True,
    aspect="auto",
    title="Correlation Heatmap of Numeric Features"
)

# (4) Genre Distribution for Track Info
if "genre" in df.columns:
    genre_counts = df["genre"].value_counts().reset_index()
    genre_counts.columns = ["genre", "count"]
    fig_genre_counts = px.pie(
        genre_counts,
        names="genre",
        values="count",
        title="Genre Distribution"
    )
else:
    fig_genre_counts = {}

# -----------------------------
# Define Dropdown Options for Dynamic Visualizations
# -----------------------------
numeric_cols = ["danceability", "loudness", "acousticness", "instrumentalness", "valence", "energy"]
initial_numeric_feature = numeric_cols[0] if numeric_cols else None

theme_cols = [
    "dating", "violence", "world/life", "night/time", "shake the audience",
    "family/gospel", "romantic", "communication", "obscene", "music",
    "movement/places", "light/visual perceptions", "family/spiritual",
    "like/girls", "sadness", "feelings"
]
initial_theme_feature = theme_cols[0] if theme_cols else None

# -----------------------------
# Function to Load Submissions from SQLite
# -----------------------------
def load_submissions():
    """Retrieve user response submissions from the SQLite database."""
    db_path = os.path.join("resources", "submissions_log.db")
    try:
        conn = sqlite3.connect(db_path)
        submissions_df = pd.read_sql_query("SELECT * FROM submissions", conn)
        conn.close()
    except Exception as e:
        print("Error loading submissions:", e)
        submissions_df = pd.DataFrame()
    return submissions_df

# -----------------------------
# Link to External CSS File (stored in the assets folder)
# -----------------------------
external_stylesheets = ['assets/custom.css']

# -----------------------------
# Dash App Layout with Scaled Banner Image
# -----------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Img(
        src='/assets/sonic_sentiments.png',
        alt='Sonic Sentiments Banner',
        style={'display': 'block', 'margin': 'auto', 'width': '18%', 'height': 'auto'}
    ),
    dcc.Tabs(id="tabs-example", value='tab-overview', children=[
        dcc.Tab(label='Overview', value='tab-overview'),
        dcc.Tab(label='Numeric Analysis', value='tab-numeric'),
        dcc.Tab(label='Thematic Analysis', value='tab-thematic'),
        dcc.Tab(label='Track Info', value='tab-trackinfo'),
        dcc.Tab(label='User Responses', value='tab-responses')
    ]),
    html.Div(id='tabs-content')
])

# -----------------------------
# Main Callback: Render Tab Content
# -----------------------------
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs-example', 'value')
)
def render_content(tab):
    if tab == 'tab-overview':
        return html.Div([
            html.H2("Overview"),
            dcc.Graph(id="overview-genre-sentiment", figure=fig_genre_sentiment),
            dcc.Graph(id="overview-release-sentiment", figure=fig_release_sentiment),
            dcc.Graph(id="overview-correlation", figure=fig_corr)
        ], style={'width': '90%', 'margin': 'auto'})

    elif tab == 'tab-numeric':
        return html.Div([
            html.H2("Numeric Analysis"),
            html.Div([
                html.Label("Select an Audio Feature:"),
                dcc.Dropdown(
                    id="numeric-feature-dropdown",
                    options=[{"label": col.capitalize(), "value": col} for col in numeric_cols],
                    value=initial_numeric_feature,
                    clearable=False,
                    style={"width": "50%"}
                )
            ], style={'padding': '20px'}),
            dcc.Graph(id="numeric-feature-histogram")
        ], style={'width': '90%', 'margin': 'auto'})

    elif tab == 'tab-thematic':
        return html.Div([
            html.H2("Thematic Analysis"),
            html.Div([
                html.Label("Select a Thematic Feature:"),
                dcc.Dropdown(
                    id="thematic-feature-dropdown",
                    options=[{"label": col.capitalize(), "value": col} for col in theme_cols],
                    value=initial_theme_feature,
                    clearable=False,
                    style={"width": "50%"}
                )
            ], style={'padding': '20px'}),
            dcc.Graph(id="thematic-feature-graph")
        ], style={'width': '90%', 'margin': 'auto'})

    elif tab == 'tab-trackinfo':
        return html.Div([
            html.H2("Track Information"),
            dcc.Graph(id="genre-distribution", figure=fig_genre_counts)
        ], style={'width': '90%', 'margin': 'auto'})

    elif tab == 'tab-responses':
        submissions_df = load_submissions()
        if submissions_df.empty:
            return html.Div([html.H2("User Responses"), html.P("No submissions available yet.")])
        
        required_cols = ['sentiment_confidence', 'textblob_score', 'predicted_genre']
        missing = [col for col in required_cols if col not in submissions_df.columns]
        if missing:
            return html.Div([html.H2("User Responses"), html.P("Submissions missing columns: " + ", ".join(missing))])
        
        submissions_df['sentiment_confidence'] = pd.to_numeric(submissions_df['sentiment_confidence'], errors='coerce')
        submissions_df['textblob_score'] = pd.to_numeric(submissions_df['textblob_score'], errors='coerce')
        submissions_df = submissions_df.reset_index().rename(columns={"index": "submission_id"})
        
        # Reshape data for a grouped bar chart comparing confidence vs. TextBlob score per submission
        df_long = submissions_df.melt(
            id_vars=["submission_id"],
            value_vars=["sentiment_confidence", "textblob_score"],
            var_name="score_type",
            value_name="score_value"
        )
        
        fig_bar = px.bar(
            df_long,
            x="submission_id",
            y="score_value",
            color="score_type",
            barmode="group",
            title="Comparison of Confidence & Sentiment per Submission",
            labels={"submission_id": "Submission ID", "score_value": "Score", "score_type": "Metric"}
        )
        # Update legend names: 'sentiment_confidence' becomes "Confidence" and 'textblob_score' becomes "Sentiment"
        fig_bar.for_each_trace(lambda t: t.update(name="Confidence" if t.name == "sentiment_confidence" else "Sentiment"))

        fig_box = px.box(
            submissions_df,
            x="predicted_genre",
            y="textblob_score",
            points="all",
            title="TextBlob Score Distribution by Predicted Genre",
            labels={"predicted_genre": "Predicted Genre", "textblob_score": "TextBlob Score"}
        )

        submissions_table = dash_table.DataTable(
            data=submissions_df.to_dict('records'),
            columns=[{"name": col, "id": col} for col in submissions_df.columns],
            page_size=10
        )

        return html.Div([
            html.H2("User Responses"),
            dcc.Graph(figure=fig_bar),
            dcc.Graph(figure=fig_box),
            html.H3("Recent Submissions"),
            submissions_table
        ], style={'width': '90%', 'margin': 'auto'})

    return html.Div("No tab selected")

# -----------------------------
# Callback for Numeric Analysis Tab
# -----------------------------
@app.callback(
    Output("numeric-feature-histogram", "figure"),
    Input("numeric-feature-dropdown", "value")
)
def update_numeric_histogram(selected_feature):
    if selected_feature:
        fig = px.histogram(
            df,
            x=selected_feature,
            nbins=30,
            title=f"Distribution of {selected_feature.capitalize()}",
            labels={selected_feature: selected_feature.capitalize()}
        )
        return fig
    return {}

# -----------------------------
# Callback for Thematic Analysis Tab
# -----------------------------
@app.callback(
    Output("thematic-feature-graph", "figure"),
    Input("thematic-feature-dropdown", "value")
)
def update_thematic_histogram(selected_feature):
    if selected_feature:
        df_temp = df.copy()
        df_temp[selected_feature] = pd.to_numeric(df_temp[selected_feature], errors='coerce')
        filtered_df = df_temp[df_temp[selected_feature] != 0]
        fig = px.histogram(
            filtered_df,
            x=selected_feature,
            nbins=30,
            title=f"Distribution of {selected_feature.capitalize()}",
            labels={selected_feature: selected_feature.capitalize()}
        )
        return fig
    return {}

if __name__ == '__main__':
    app.run(debug=True)
