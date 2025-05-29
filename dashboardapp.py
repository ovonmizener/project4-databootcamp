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

# If available, extract the release year from 'release_date'
if "release_date" in df.columns:
    df["release_year"] = pd.to_datetime(df["release_date"], errors='coerce').dt.year

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
# For Numeric Analysis (audio features)
numeric_cols = ["danceability", "loudness", "acousticness", "instrumentalness", "valence", "energy"]
initial_numeric_feature = numeric_cols[0] if numeric_cols else None

# For Thematic Analysis
theme_cols = ["dating", "violence", "world/life", "night/time", "shake the audience",
              "family/gospel", "romantic", "communication", "obscene", "music",
              "movement/places", "light/visual perceptions", "family/spiritual",
              "like/girls", "sadness", "feelings"]
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
# Link to External CSS File (located in the assets folder)
# -----------------------------
# Dash automatically looks in the assets folder for any CSS files.
# You can also explicitly specify external stylesheets as shown below.
external_stylesheets = ['assets/custom.css']

# -----------------------------
# Dash App Layout with Tabs
# -----------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("Music Dataset Trends Dashboard", style={'textAlign': 'center'}),
    dcc.Tabs(id="tabs-example", value='tab-overview', children=[
        dcc.Tab(label='Overview', value='tab-overview'),
        dcc.Tab(label='Numeric Analysis', value='tab-numeric'),
        dcc.Tab(label='Thematic Analysis', value='tab-thematic'),
        dcc.Tab(label='Track Info', value='tab-trackinfo'),
        dcc.Tab(label='User Responses', value='tab-responses')
    ]),
    html.Div(id='tabs-content')
])

# Callback to render the content for each tab dynamically.
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
            response_content = html.Div([
                html.H2("User Responses"),
                html.P("No submissions available yet.")
            ])
        else:
            # Create a histogram of TextBlob scores from user submissions.
            # Note: Use "textblob_score" as that is the actual column name.
            fig_tb_hist = px.histogram(
                submissions_df, 
                x="textblob_score", 
                nbins=20, 
                title="Distribution of User TextBlob Scores"
            )
            # Create a data table for recent submissions.
            submissions_table = dash_table.DataTable(
                data=submissions_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in submissions_df.columns],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            )
            response_content = html.Div([
                html.H2("User Responses"),
                dcc.Graph(figure=fig_tb_hist),
                html.H3("Recent Submissions"),
                submissions_table
            ])
        return html.Div(response_content, style={'width': '90%', 'margin': 'auto'})
    
    return html.Div("No tab selected")

# -----------------------------
# Callbacks for Dynamic Graphs in Numeric and Thematic Analysis Tabs
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

@app.callback(
    Output("thematic-feature-graph", "figure"),
    Input("thematic-feature-dropdown", "value")
)
def update_thematic_histogram(selected_feature):
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

if __name__ == '__main__':
    app.run(debug=True)
