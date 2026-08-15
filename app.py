"""
Movie Recommender System - Streamlit App
==========================================
Content-based movie recommender using TF-IDF on title, overview+tagline, and genres.
Based on the analysis/training notebook (final__project__5_.ipynb).

Expected data files (place them in a folder named `data/` next to this script,
or change DATA_DIR below):
    - movies_metadata.csv
    - ratings_small.csv
    - links_small.csv

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import ast
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_DIR = Path(".")
MOVIES_PATH = DATA_DIR / "movies_metadata.csv"
RATINGS_PATH = DATA_DIR / "ratings_small.csv"
LINKS_PATH = DATA_DIR / "links_small.csv"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def extract_names(field_str):
    """Extract the 'name' values from a stringified list of dicts, e.g. genres."""
    try:
        items_list = ast.literal_eval(field_str)
        return " ".join([item["name"] for item in items_list])
    except Exception:
        return ""


@st.cache_data(show_spinner="Loading and cleaning data...")
def load_and_clean_data():
    movies = pd.read_csv(MOVIES_PATH, low_memory=False)
    ratings = pd.read_csv(RATINGS_PATH)
    links = pd.read_csv(LINKS_PATH)

    # Keep rows with a valid numeric id
    movies = movies[movies["id"].astype(str).str.isnumeric()]
    movies["id"] = movies["id"].astype(int)
    movies["budget"] = pd.to_numeric(movies["budget"], errors="coerce")
    movies["popularity"] = pd.to_numeric(movies["popularity"], errors="coerce")
    movies["revenue"] = pd.to_numeric(movies["revenue"], errors="coerce")

    drop_cols = [
        c
        for c in [
            "belongs_to_collection",
            "homepage",
            "spoken_languages",
            "original_title",
            "poster_path",
        ]
        if c in movies.columns
    ]
    movies.drop(columns=drop_cols, inplace=True)
    movies.drop_duplicates(inplace=True, keep="first")

    movies["tagline"] = movies["tagline"].fillna("")
    movies["overview"] = movies["overview"].fillna("")
    movies.dropna(inplace=True)

    movies["production_countries"] = movies["production_countries"].apply(extract_names)
    movies["production_companies"] = movies["production_companies"].apply(extract_names)
    movies["genres"] = movies["genres"].apply(extract_names)

    # Model-ready subset
    data_model = movies[
        [
            "id",
            "imdb_id",
            "genres",
            "original_language",
            "overview",
            "title",
            "production_countries",
            "production_companies",
            "runtime",
            "tagline",
        ]
    ].copy()
    data_model = data_model.reset_index(drop=True)

    # Fill numeric gaps on the full movies frame (kept for consistency with notebook)
    movies["runtime"] = movies["runtime"].replace(0, np.nan)
    movies["budget"] = movies["budget"].replace(0, np.nan)
    movies["revenue"] = movies["revenue"].replace(0, np.nan)
    movies["runtime"] = movies["runtime"].fillna(movies["runtime"].median())
    movies["budget"] = movies["budget"].fillna(movies["budget"].mean())
    movies["revenue"] = movies["revenue"].fillna(movies["revenue"].mean())

    # Drop rows with empty text fields needed for the model
    data_model = data_model[data_model["genres"] != ""]
    data_model = data_model[data_model["overview"] != ""]
    data_model = data_model[data_model["production_companies"] != ""]
    data_model = data_model[data_model["production_countries"] != ""]
    data_model = data_model[data_model["tagline"] != ""]
    data_model = data_model.reset_index(drop=True)

    return movies, ratings, links, data_model


@st.cache_resource(show_spinner="Training TF-IDF model...")
def train_model(data_model: pd.DataFrame):
    text_content = data_model["overview"] + " " + data_model["tagline"]
    genre_content = data_model["genres"]

    tfidf_title = TfidfVectorizer(stop_words="english")
    vec_title = tfidf_title.fit_transform(data_model["title"].values.astype("U"))

    tfidf_text = TfidfVectorizer(max_features=40000, stop_words="english")
    vec_text = tfidf_text.fit_transform(text_content.values.astype("U"))

    tfidf_genre = TfidfVectorizer()
    vec_genre = tfidf_genre.fit_transform(genre_content.values.astype("U"))

    return vec_title, vec_text, vec_genre


def recommend(data_model, vec_title, vec_text, vec_genre, movie_title, top_n=5,
              w_title=0.5, w_text=0.3, w_genre=0.2):
    matches = data_model[data_model["title"] == movie_title]
    if matches.empty:
        return []
    index = matches.index[0]

    sim_title = cosine_similarity(vec_title[index], vec_title).flatten()
    sim_text = cosine_similarity(vec_text[index], vec_text).flatten()
    sim_genre = cosine_similarity(vec_genre[index], vec_genre).flatten()

    sim = w_title * sim_title + w_text * sim_text + w_genre * sim_genre

    distance = sorted(list(enumerate(sim)), reverse=True, key=lambda x: x[1])
    results = []
    for i, s in distance[1 : top_n + 1]:
        row = data_model.iloc[i]
        results.append({"title": row["title"], "score": float(s), "genres": row["genres"],
                         "overview": row["overview"]})
    return results


# --------------------------------------------------------------------------- #
# App UI
# --------------------------------------------------------------------------- #
def main():
    st.title("🎬 Movie Recommender System")
    st.caption("Content-based recommendations using TF-IDF on title, overview, tagline and genres.")

    if not MOVIES_PATH.exists() or not RATINGS_PATH.exists() or not LINKS_PATH.exists():
        st.error(
            "Data files not found. Please put `movies_metadata.csv`, `ratings_small.csv` "
            f"and `links_small.csv` inside a `{DATA_DIR}/` folder next to this app."
        )
        st.stop()

    movies, ratings, links, data_model = load_and_clean_data()
    vec_title, vec_text, vec_genre = train_model(data_model)

    with st.sidebar:
        st.header("⚙️ Settings")
        top_n = st.slider("Number of recommendations", min_value=1, max_value=20, value=6)
        st.subheader("Similarity weights")
        w_title = st.slider("Title weight", 0.0, 1.0, 0.5, 0.05)
        w_text = st.slider("Overview/Tagline weight", 0.0, 1.0, 0.3, 0.05)
        w_genre = st.slider("Genre weight", 0.0, 1.0, 0.2, 0.05)

    titles = sorted(data_model["title"].dropna().unique().tolist())
    default_index = titles.index("Toy Story") if "Toy Story" in titles else 0
    movie_title = st.selectbox("Pick a movie you like:", titles, index=default_index)

    if st.button("Get Recommendations", type="primary"):
        results = recommend(
            data_model, vec_title, vec_text, vec_genre, movie_title,
            top_n=top_n, w_title=w_title, w_text=w_text, w_genre=w_genre,
        )

        if not results:
            st.warning("No recommendations found for this movie.")
        else:
            st.subheader(f"Because you liked '{movie_title}':")
            for r in results:
                with st.container(border=True):
                    st.markdown(f"**{r['title']}**  \n*Genres:* {r['genres']}  \n*Similarity score:* {r['score']:.3f}")
                    with st.expander("Overview"):
                        st.write(r["overview"])

    with st.expander("📊 Dataset info"):
        st.write(f"Movies in model: {len(data_model)}")
        st.write(f"Ratings: {len(ratings)}")
        st.dataframe(data_model.head(20))


if __name__ == "__main__":
    main()
