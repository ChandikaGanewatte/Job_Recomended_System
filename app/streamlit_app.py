import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import PyPDF2

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AI Job Dashboard",
    page_icon="💼",
    layout="wide"
)

# -------------------------
# CUSTOM HEADER
# -------------------------
st.markdown("""
    <style>
        .main-title {
            font-size:40px;
            font-weight:700;
            color:#1f4e79;
        }
        .sub-title {
            font-size:16px;
            color:gray;
        }
        .card {
            padding:15px;
            border-radius:12px;
            border:1px solid #ddd;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
            margin-bottom:10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💼 AI Career Recommendation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload your resume and get AI-powered job matches instantly</div>', unsafe_allow_html=True)

st.divider()

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv('../data/processed/processed_jobs.csv')

jobs_df = load_data()

# -------------------------
# MODEL
# -------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

@st.cache_data
def compute_embeddings(texts):
    return model.encode(texts, show_progress_bar=False)

job_emb = compute_embeddings(jobs_df['text'].tolist())

# -------------------------
# PDF TEXT EXTRACTION
# -------------------------
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# -------------------------
# SIDEBAR FILTERS
# -------------------------
st.sidebar.header("🎛 Filters")

top_n = st.sidebar.slider("Top recommendations", 3, 10, 5)

location_filter = st.sidebar.text_input("Filter by location (optional)")
exp_filter = st.sidebar.text_input("Filter by experience (optional)")

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3 = st.tabs(["📄 Upload Resume", "🎯 Recommendations", "📊 Insights"])

resume_text = ""

# -------------------------
# TAB 1 - INPUT
# -------------------------
with tab1:
    st.subheader("Upload your Resume")

    uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])
    manual_text = st.text_area("OR paste resume text", height=150)

    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success("Resume uploaded successfully!")
        st.text(resume_text[:800])

    elif manual_text:
        resume_text = manual_text

# -------------------------
# TAB 2 - RESULTS
# -------------------------
with tab2:

    if st.button("🚀 Generate Recommendations"):

        if resume_text.strip() == "":
            st.warning("Please upload or enter resume first")
        else:

            with st.spinner("AI is analyzing your profile..."):

                resume_emb = model.encode([resume_text])
                sim = cosine_similarity(resume_emb, job_emb)[0]

                jobs_df["score"] = sim

                filtered = jobs_df.copy()

                if location_filter:
                    filtered = filtered[filtered["location"].str.contains(location_filter, case=False, na=False)]

                if exp_filter:
                    filtered = filtered[filtered["formatted_experience_level"].str.contains(exp_filter, case=False, na=False)]

                filtered = filtered.sort_values("score", ascending=False).head(top_n)

                # ---------------- KPI METRICS ----------------
                col1, col2, col3 = st.columns(3)

                col1.metric("Top Match Score", f"{filtered['score'].max():.2f}")
                col2.metric("Jobs Found", len(filtered))
                col3.metric("AI Model", "SBERT")

                st.divider()

                # ---------------- JOB CARDS ----------------
                for _, job in filtered.iterrows():

                    st.markdown(f"""
                        <div class="card">
                            <h3>💼 {job['title']}</h3>
                            <p>📍 <b>Location:</b> {job.get('location','N/A')}</p>
                            <p>📊 <b>Experience:</b> {job.get('formatted_experience_level','N/A')}</p>
                            <p>💰 <b>Salary:</b> {job.get('normalized_salary','N/A')}</p>
                            <p>🎯 <b>Match Score:</b> {job['score']:.2f}</p>
                        </div>
                    """, unsafe_allow_html=True)

# -------------------------
# TAB 3 - INSIGHTS
# -------------------------
with tab3:
    st.subheader("📊 System Insights")

    st.write("Top job categories in dataset:")

    if "title" in jobs_df.columns:
        st.bar_chart(jobs_df["title"].value_counts().head(10))