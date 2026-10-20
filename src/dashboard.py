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


def pipeline_key() -> tuple:
    """Identifies the currently selected dataset, for caching derived results."""
    return (current_source(), st.session_state.get("max_rows"), st.session_state.get("upload_key"))


@st.cache_resource(show_spinner="Computing SHAP values...")
def _shap_cache(key, _res):
    from shap_explainer import shap_values
    return shap_values(_res.clf, _res.df)


def missing_package_stop(err: ModuleNotFoundError):
    """A package from requirements.txt is missing -- usually because the
    `streamlit` command belongs to a different Python than the one the
    requirements were installed into."""
    st.error(
        f"Missing Python package `{err.name}`. Install the requirements into the same Python "
        "that runs Streamlit:\n\n```\npython -m pip install -r requirements.txt\n"
        "python -m streamlit run app.py\n```"
    )
    st.stop()


def get_shap_values(res: PipelineResult):
    """SHAP contributions (n_reviews x 6) for the loaded dataset, cached."""
    try:
        return _shap_cache(pipeline_key() + (len(res.df),), res)
    except ModuleNotFoundError as err:
        missing_package_stop(err)


@st.cache_resource(show_spinner="Loading text model for LIME...")
def _text_model_cache(key, _res):
    from lime_explainer import load_text_model
    train = _res.df.iloc[_res.train_idx]
    label = "ground_truth_label" if "ground_truth_label" in train.columns else None
    if label is None:
        # unlabeled data: fall back to the model's own predictions as labels
        labels = _res.clf.predict(train[_res.clf.feature_names_in_])
        return load_text_model(train["clean_text"], labels)
    return load_text_model(train["clean_text"], train[label])


def get_text_model(res: PipelineResult):
    try:
        return _text_model_cache(pipeline_key() + (len(res.df),), res)
    except ModuleNotFoundError as err:
        missing_package_stop(err)
