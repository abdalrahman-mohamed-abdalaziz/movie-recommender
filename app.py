import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)

    # keep only rows with a valid numeric id
    movies = movies[movies['id'].astype(str).str.isnumeric()]
    movies['id'] = movies['id'].astype(int)

    movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce')
    movies['popularity'] = pd.to_numeric(movies['popularity'], errors='coerce')
    movies['revenue'] = pd.to_numeric(movies['revenue'], errors='coerce')

    drop_cols = ['belongs_to_collection', 'homepage', 'spoken_languages',
                 'original_title', 'poster_path']
    movies.drop(columns=[c for c in drop_cols if c in movies.columns], inplace=True)
    movies.drop_duplicates(inplace=True, keep='first')

    movies['tagline'] = movies['tagline'].fillna('')
    movies['overview'] = movies['overview'].fillna('')
    movies.dropna(inplace=True)

    def extract_names(field_str):
        try:
            items_list = ast.literal_eval(field_str)
            return " ".join([item['name'] for item in items_list])
        except Exception:
            return ""

    movies['production_countries'] = movies['production_countries'].apply(extract_names)
    movies['production_companies'] = movies['production_companies'].apply(extract_names)
    movies['genres'] = movies['genres'].apply(extract_names)

    # keep only the top 10,000 movies by popularity to limit dataset size
    movies = movies.sort_values('popularity', ascending=False).head(10000)

    data_model = movies[['id', 'imdb_id', 'genres', 'original_language', 'overview',
                          'title', 'production_countries', 'production_companies',
                          'runtime', 'tagline']].copy()
    data_model = data_model.reset_index(drop=True)

    data_model = data_model[data_model['genres'] != '']
    data_model = data_model[data_model['overview'] != '']
    data_model = data_model[data_model['tagline'] != '']

    # avoid duplicate titles confusing the selectbox / lookup
    data_model = data_model.drop_duplicates(subset='title').reset_index(drop=True)

    return data_model


@st.cache_resource(show_spinner=False)
def build_model(data_model: pd.DataFrame):
    text_content = data_model['overview'] + ' ' + data_model['tagline']
    genre_content = data_model['genres']

    tfidf_title = TfidfVectorizer(stop_words='english')
    vec_title = tfidf_title.fit_transform(data_model['title'].values.astype('U'))

    tfidf_text = TfidfVectorizer(max_features=40000, stop_words='english')
    vec_text = tfidf_text.fit_transform(text_content.values.astype('U'))

    tfidf_genre = TfidfVectorizer()
    vec_genre = tfidf_genre.fit_transform(genre_content.values.astype('U'))

    return vec_title, vec_text, vec_genre


def recommend(movie_title, data_model, vec_title, vec_text, vec_genre,
              top_n=5, w_title=0.5, w_text=0.3, w_genre=0.2):
    matches = data_model[data_model['title'] == movie_title]
    if matches.empty:
        return pd.DataFrame()
    index = matches.index[0]

    sim_title = cosine_similarity(vec_title[index], vec_title).flatten()
    sim_text = cosine_similarity(vec_text[index], vec_text).flatten()
    sim_genre = cosine_similarity(vec_genre[index], vec_genre).flatten()

    sim = w_title * sim_title + w_text * sim_text + w_genre * sim_genre

    ranked = sorted(list(enumerate(sim)), reverse=True, key=lambda x: x[1])
    top_indices = [i for i, s in ranked[1:top_n + 1]]
    return data_model.iloc[top_indices][['title', 'genres', 'overview', 'tagline']]


def main():
    st.title("🎬 Movie Recommender System")
    st.caption("نظام توصية أفلام Content-Based بناءً على العنوان، القصة، والنوع")

    try:
        with st.spinner("جاري تحميل البيانات وتجهيز النموذج..."):
            data_model = load_and_prepare_data()
            vec_title, vec_text, vec_genre = build_model(data_model)
    except FileNotFoundError:
        st.error(
            "لم يتم العثور على ملف movies_metadata.csv.\n\n"
            "لازم ترفع الملف في نفس المجلد اللي فيه app.py قبل تشغيل التطبيق."
        )
        st.stop()

    top_n = 10
    w_title = 0.15
    w_text = 0.55
    w_genre = 0.3

    movie_list = sorted(data_model['title'].dropna().unique())
    selected_movie = st.selectbox("اختر فيلم:", movie_list)

    if st.button("Recommend other movies 🎯", type="primary"):
        results = recommend(
            selected_movie, data_model, vec_title, vec_text, vec_genre,
            top_n=top_n, w_title=w_title, w_text=w_text, w_genre=w_genre
        )
        if results.empty:
            st.warning("لم يتم العثور على الفيلم في قاعدة البيانات.")
        else:
            st.subheader(f"أفلام مشابهة لـ: {selected_movie}")
            for _, row in results.iterrows():
                with st.expander(f"🎬 {row['title']}"):
                    st.write(f"**النوع:** {row['genres']}")
                    st.write(f"**الوصف:** {row['overview']}")
                    if row['tagline']:
                        st.write(f"**Tagline:** {row['tagline']}")


if __name__ == "__main__":
    main()
