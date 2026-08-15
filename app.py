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

    # 1. تصفية ومعالجة المعرفات والأرقام
    movies = movies[movies["id"].str.isnumeric()].copy()
    movies["id"] = movies["id"].astype(int)

    # 2. إزالة التكرارات
    movies.drop_duplicates(subset=["id"], inplace=True, keep="first")

    # 3. تجهيز النصوص وتحضير الحقول
    movies["tagline"] = movies["tagline"].fillna("")
    movies["overview"] = movies["overview"].fillna("")

    movies["genres"] = movies["genres"].apply(extract_names)
    movies["production_companies"] = movies["production_companies"].apply(extract_names)
    movies["production_countries"] = movies["production_countries"].apply(extract_names)

    # 4. بناء داتا الموديل وإعادة ضبط الفهرس لضمان التطابق مع TF-IDF
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
            "tagline",
        ]
    ].copy()

    # إزالة الصفوف التي تفتقر للعنوان أو الوصف لتجنب الضوضاء
    data_model = data_model[data_model["title"].notnull() & (data_model["overview"] != "")]
    
    # خطوة حاسمة: إعادة ضبط الفهرس لتطابق المصفوفة 1:1
    data_model.reset_index(drop=True, inplace=True)

    # 5. بناء حقل المحتوى النصي كما في Colab
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


def recommend(data_model, vector, movie_title, top_n=6):
    matches = data_model[data_model["title"] == movie_title]
    if matches.empty:
        return []
    
    # الحصول على موقع الصف الفعلي داخل الـ DataFrame
    index = matches.index[0]
    
    # حساب أرقام التشابه
    sim_scores = cosine_similarity(vector[index], vector).flatten()
    
    # ترتيب الأفلام حسب أعلى درجة تشابه (مع استبعاد الفيلم نفسه)
    distance = sorted(list(enumerate(sim_scores)), reverse=True, key=lambda x: x[1])
    top_indices = [i for i, _ in distance[1 : top_n + 1]]
    
    return data_model.iloc[top_indices]["title"].tolist()


# ---------------- الواجهة ----------------

st.title("🎬 نظام توصية الأفلام")
st.caption("توصيات دقيقة متطابقة مع نتائج كود Colab")

try:
    with st.spinner("جاري التحميل وتجهيز النموذج..."):
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
        recommendations = recommend(data_model, vector, selected_movie, top_n)

        if not recommendations:
            st.warning("لم يتم العثور على توصيات للفيلم المختار.")
        else:
            st.subheader(f"التوصيات لـ {selected_movie}:")
            st.code(str(recommendations), language="python")

            for idx, movie in enumerate(recommendations, 1):
                st.write(f"**{idx}.** {movie}")

    st.divider()
    st.caption(f"عدد الأفلام المتاحة: {len(data_model):,} فيلم")

except Exception as e:
    st.error(f"حدث خطأ: {e}")
