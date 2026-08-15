import ast

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_searchbox import st_searchbox

st.set_page_config(page_title="نظام توصية الأفلام", page_icon="🎬", layout="centered")


def extract_names(field_str):
    try:
        items_list = ast.literal_eval(field_str)
        return " ".join([item["name"] for item in items_list])
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def load_and_clean_data():
    ratings = pd.read_csv("ratings_small.csv")
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)

    movies = movies[movies["id"].str.isnumeric()]
    movies["id"] = movies["id"].astype(int)
    movies["budget"] = pd.to_numeric(movies["budget"], errors="coerce")
    movies["popularity"] = pd.to_numeric(movies["popularity"], errors="coerce")
    movies["revenue"] = pd.to_numeric(movies["revenue"], errors="coerce")

    movies.drop(
        ["belongs_to_collection", "homepage", "spoken_languages", "original_title", "poster_path"],
        inplace=True,
        axis=1,
        errors="ignore",
    )
    movies.drop_duplicates(inplace=True, keep="first")

    movies["tagline"] = movies["tagline"].fillna("")
    movies["overview"] = movies["overview"].fillna("")
    movies.dropna(inplace=True)

    movies["production_countries"] = movies["production_countries"].apply(extract_names)
    movies["production_companies"] = movies["production_companies"].apply(extract_names)
    movies["genres"] = movies["genres"].apply(extract_names)

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

    movies["runtime"] = movies["runtime"].replace(0, np.nan)
    movies["budget"] = movies["budget"].replace(0, np.nan)
    movies["revenue"] = movies["revenue"].replace(0, np.nan)
    movies["runtime"] = movies["runtime"].fillna(movies["runtime"].median())
    movies["budget"] = movies["budget"].fillna(movies["budget"].mean())
    movies["revenue"] = movies["revenue"].fillna(movies["revenue"].mean())

    movies = movies[movies["genres"] != ""]
    movies = movies[movies["overview"] != ""]
    movies = movies[movies["production_companies"] != ""]
    movies = movies[movies["production_countries"] != ""]
    movies = movies[movies["tagline"] != ""]

    outlier_cols = ["popularity", "vote_count", "vote_average"]
    for col in outlier_cols:
        q1 = movies[col].quantile(0.25)
        q3 = movies[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        movies = movies[(movies[col] >= lower) & (movies[col] <= upper)]

    data_model["text_content"] = data_model["overview"] + " " + data_model["tagline"]

    return movies, data_model, ratings


@st.cache_resource(show_spinner=False)
def build_model(data_model):
    tfidf_text = TfidfVectorizer(max_features=40000, stop_words="english")
    vec_text = tfidf_text.fit_transform(data_model["text_content"].values.astype("U"))

    tfidf_genre = TfidfVectorizer()
    vec_genre = tfidf_genre.fit_transform(data_model["genres"].values.astype("U"))

    return vec_text, vec_genre


def recommend(data_model, vec_text, vec_genre, movie_title, top_n=5, w_text=0.5, w_genre=0.5):
    matches = data_model[data_model["title"] == movie_title]
    if matches.empty:
        return pd.DataFrame()
    index = matches.index[0]
    sim_text = cosine_similarity(vec_text[index], vec_text).flatten()
    sim_genre = cosine_similarity(vec_genre[index], vec_genre).flatten()
    sim_scores = w_text * sim_text + w_genre * sim_genre
    distance = sorted(list(enumerate(sim_scores)), reverse=True, key=lambda x: x[1])
    top_indices = [i for i, _ in distance[1 : top_n + 1]]
    result = data_model.iloc[top_indices][["title", "genres", "overview"]].copy()
    result["similarity"] = [sim_scores[i] for i in top_indices]
    return result


# ---------------- UI ----------------

st.title("🎬 نظام توصية الأفلام")
st.caption("اختار فيلم عجبك وهنقترحلك أفلام شبهه بناءً على القصة والنوع")

with st.spinner("بنجهز الداتا والموديل... (بيحصل مرة واحدة بس)"):
    movies, data_model, ratings = load_and_clean_data()
    vec_text, vec_genre = build_model(data_model)

titles = sorted(data_model["title"].unique().tolist())

selected_movie = st.selectbox("اختار فيلم:", titles, index=titles.index("Toy Story") if "Toy Story" in titles else 0)
top_n = st.slider("عدد التوصيات:", min_value=3, max_value=15, value=5)


genre_weight = st.slider(
    "وزن النوع مقابل القصة:", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
    help="0 = التوصية على أساس القصة بس - 1 = التوصية على أساس النوع بس",
)

if st.button("وريني توصيات", type="primary"):
    if not selected_movie:
        st.warning("اختار فيلم الأول من مربع البحث فوق.")
    else:
        with st.spinner("بنحسب أقرب الأفلام..."):
            results = recommend(
                data_model, vec_text, vec_genre, selected_movie,
                top_n=top_n, w_text=1 - genre_weight, w_genre=genre_weight,
            )

        if results.empty:
            st.warning("مفيش بيانات كفاية عن الفيلم ده.")
        else:
            st.subheader(f"أفلام شبه {selected_movie}")
            for _, row in results.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['title']}**  \n*{row['genres']}*")
                    st.caption(row["overview"][:300] + ("..." if len(row["overview"]) > 300 else ""))
                    st.progress(
                        min(max(float(row["similarity"]), 0.0), 1.0),
                        text=f"نسبة التشابه: {row['similarity']:.0%}",
                    )

st.divider()
st.caption(f"عدد الأفلام المتاحة للتوصية: {len(data_model):,} فيلم")
