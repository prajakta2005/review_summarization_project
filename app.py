"""
app.py
------
Home page of the Streamlit dashboard (multipage app: every file in
pages/ is one module / feature).

Run with:  streamlit run app.py

Workflow shown on screen matches the patent's end-to-end pipeline:
Upload -> Preprocess -> Extract Features -> Score Credibility ->
Explain -> Credibility-Aware Summarize -> Dashboard/Output

Home picks the data source and stores it in st.session_state; every
page then gets the same cached pipeline result via require_pipeline().
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard import SOURCE_LABELS, require_pipeline
from explainability import global_feature_importance

st.set_page_config(page_title="Review Credibility Intelligence", layout="wide")
st.title("🔍 Explainable Consumer Review Intelligence System")
st.caption("Credibility assessment + explainability + credibility-aware summarization")

# --- Data source (stored in session_state so every page uses it) ---
st.sidebar.header("1. Load Reviews")
sources = list(SOURCE_LABELS)
source = st.sidebar.radio(
    "Data source", sources,
    index=sources.index(st.session_state.get("data_source", "sample")),
    format_func=SOURCE_LABELS.get,
)

if source == "upload":
    uploaded = st.sidebar.file_uploader(
        "Upload a CSV -- either our schema (review_id, product_id, reviewer_id, rating, "
        "timestamp, review_text) OR the raw Kaggle fake-reviews-dataset schema "
        "(category, rating, label, text_) -- both are auto-detected.",
        type="csv"
    )
    if uploaded is not None:
        st.session_state["uploaded_df"] = pd.read_csv(uploaded)
        st.session_state["upload_key"] = f"{uploaded.name}-{uploaded.size}"
    elif st.session_state.get("uploaded_df") is None:
        st.info("Upload a CSV to begin, or pick another data source.")
        st.stop()

# --- Row limit: the full 40k Kaggle set takes ~11s on first load (then
# it's cached), so "all" is the default; smaller samples are stratified.
ROW_OPTIONS = [1000, 3000, 10000, 20000, "all"]
max_rows = "all"
if source != "sample":
    max_rows = st.sidebar.select_slider(
        "Max rows", ROW_OPTIONS,
        value=st.session_state.get("max_rows") or "all",
        help="Stratified down-sampling for faster experiments. 'all' uses every review.",
    )
st.session_state["data_source"] = source
st.session_state["max_rows"] = None if max_rows == "all" else max_rows

res = require_pipeline()
explained, clf = res.explained, res.clf

# --- Dashboard summary metrics ---
st.header("Dashboard Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews Analysed", len(explained))
col2.metric("Average Credibility Score", f"{explained['credibility_score'].mean():.1f}")
dist = explained["credibility_label"].value_counts()
col3.metric("High Credibility", int(dist.get("High", 0)))
col4.metric("Low Credibility", int(dist.get("Low", 0)))

c1, c2 = st.columns(2)
with c1:
    fig = px.pie(explained, names="credibility_label", title="Credibility Distribution",
                 color="credibility_label",
                 color_discrete_map={"High": "#2ecc71", "Medium": "#f1c40f", "Low": "#e74c3c"})
    st.plotly_chart(fig, use_container_width=True)
with c2:
    importance = global_feature_importance(clf).reset_index()
    importance.columns = ["feature", "importance"]
    fig2 = px.bar(importance, x="importance", y="feature", orientation="h",
                  title="Which signals matter most (global feature importance)")
    st.plotly_chart(fig2, use_container_width=True)

st.info("Use the pages in the sidebar to explore each module. "
        "**Review Explorer** has product summaries and per-review explanations.")
