"""
dashboard.py
------------
Streamlit helpers shared by app.py (Home) and every page in pages/.

The data source is chosen on Home and stored in st.session_state,
because in a multipage app widgets on one page don't exist on the
others. Every page calls `require_pipeline()` to get the cached
PipelineResult for whatever is currently selected.
"""

import streamlit as st

from pipeline import load_raw, run_pipeline, PipelineResult
from results_io import load_result

SOURCE_LABELS = {
    "sample": "Bundled sample (15 reviews)",
    "kaggle_full": "Kaggle fake-reviews dataset (≈40k reviews)",
    "upload": "Upload a CSV",
}


@st.cache_resource(show_spinner="Running pipeline: preprocessing -> features -> credibility model -> explainability...")
def get_pipeline(source: str, max_rows: int = None, upload_key: str = None,
                 _uploaded_df=None) -> PipelineResult:
    # upload_key identifies the uploaded file for caching; the dataframe
    # itself is passed with a leading underscore so streamlit doesn't hash it.
    raw = load_raw(source, uploaded_df=_uploaded_df, max_rows=max_rows)
    return run_pipeline(raw)


def current_source() -> str:
    return st.session_state.get("data_source", "sample")


def require_pipeline() -> PipelineResult:
    """Top of every page: returns the PipelineResult for the data source
    picked on Home (defaults to the bundled sample)."""
    source = current_source()
    max_rows = st.session_state.get("max_rows")
    uploaded_df = st.session_state.get("uploaded_df")
    upload_key = st.session_state.get("upload_key")
    if source == "upload" and uploaded_df is None:
        source = "sample"
    try:
        res = get_pipeline(source, max_rows, upload_key if source == "upload" else None,
                           _uploaded_df=uploaded_df if source == "upload" else None)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.caption(f"Dataset: **{SOURCE_LABELS[source]}** · {len(res.explained):,} reviews "
               f"· model trained on {len(res.train_idx):,}, tested on {len(res.test_idx):,} "
               f"· change it on the Home page")
    return res


def show_result_or_hint(name: str, script: str):
    """Loads results/<name>.json, or explains how to generate it."""
    result = load_result(name)
    if result is None:
        st.info(f"No results yet for **{name}**. Generate them with:\n\n```\npython {script}\n```")
    return result
