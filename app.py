import ast
import os
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="نظام توصية الأفلام", page_icon="🎬", layout="centered")

MOVIES_URL = "https://raw.githubusercontent.com/Rouby2004/Movie-Recommendation-System/main/movies_metadata.csv"
RATINGS_URL = "https://raw.githubusercontent.com/Rouby2004/Movie-Recommendation-System/main/ratings_small.csv"

MOVIES_FILE = "movies_metadata.csv"
RATINGS_FILE = "ratings_small.csv"


def download_file_if_missing(url, file_path):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with st.spinner(f"جاري تنزيل ملف البيانات {file_path}..."):
            urllib.request.urlretrieve(url, file_path)


def extract_names(field_str):
    try:
        items_list = ast.literal_eval(field_str)
        return " ".join([item["name"] for item in items_list])
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def load_and_clean_data():
    download_file_if_missing(MOVIES_URL, MOVIES_FILE)
    download_file_if_missing(RATINGS_URL, RATINGS_FILE)

    ratings = pd.read_csv(RATINGS_FILE)
    movies = pd.read_csv(MOVIES_FILE, low_memory=False)

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

    data_model["content"] = (
        data_model["overview"]
        + " "
        + ((data_model["genres"] + " ") * 3)
        + data_model["tagline"]
    )

    return movies, data_model, ratings


@st.cache_resource(show_spinner=False)
def build_model(data_model):
    tfidf = TfidfVectorizer(max_features=40000, stop_words="english")
    vector = tfidf.fit_transform(data_model["content"].values.astype("U"))
    return tfidf, vector


# دالة التوصية بنفس طريقة Colab وترجع قائمة أسماء (List of titles)
def recommend(movie_title, top_n=6):
    matches = data_model[data_model["title"] == movie_title]
    if matches.empty:
        return []
    index = matches.index[0]
    sim_scores = cosine_similarity(vector[index], vector).flatten()
    distance = sorted(list(enumerate(sim_scores)), reverse=True, key=lambda x: x[1])
    top_indices = [i for i, _ in distance[1 : top_n + 1]]
    return data_model.iloc[top_indices]["title"].tolist()


# ---------------- الواجهة ----------------

st.title("🎬 نظام توصية الأفلام")
st.caption("اختار فيلم عجبك وهنقترحلك أفلام شبهه بناءً على القصة والنوع")

try:
    with st.spinner("بنجهز الداتا والموديل..."):
        movies, data_model, ratings = load_and_clean_data()
        tfidf, vector = build_model(data_model)

    titles = sorted(data_model["title"].unique().tolist())

    selected_movie = st.selectbox(
        "اختار فيلم:",
        titles,
        index=titles.index("Toy Story") if "Toy Story" in titles else 0,
    )
    top_n = st.number_input("عدد التوصيات:", min_value=1, max_value=20, value=6)

    if st.button("وريني توصيات", type="primary"):
        recommendations = recommend(selected_movie, top_n)

        if not recommendations:
            st.warning("مفيش بيانات كفاية عن الفيلم ده.")
        else:
            st.subheader(f"التوصيات لـ {selected_movie}:")
            # عرض النتيجة على شكل القائمة النصية المطابقة لـ Colab
            st.code(str(recommendations), language="python")

            # عرض النتائج بشكل قائمة منسقة
            for idx, movie in enumerate(recommendations, 1):
                st.write(f"**{idx}.** {movie}")

    st.divider()
    st.caption(f"عدد الأفلام المتاحة للتوصية: {len(data_model):,} فيلم")

except Exception as e:
    st.error(f"حدث خطأ أثناء تحميل البيانات: {e}")
