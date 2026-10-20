"""
pipeline.py
-----------
The end-to-end flow shown on the dashboard, in one reusable function:
Load -> Preprocess -> Extract Features -> Score Credibility -> Explain.

No streamlit imports here on purpose: the experiment scripts in
experiments/ use the exact same code path as the dashboard.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from preprocessing import load_reviews, preprocess
from features import extract_features
from credibility_model import train_model, score_reviews, labeled_split
from explainability import explain_dataframe
from kaggle_adapter import adapt_kaggle_dataframe, _stratified_sample

DATA_SOURCES = {
    "sample": "data/sample_reviews.csv",
    "kaggle_full": "data/fake reviews dataset.csv",
}

KAGGLE_COLS = {"category", "rating", "label", "text_"}
OUR_COLS = {"review_id", "product_id", "reviewer_id", "rating", "timestamp", "review_text"}


def adapt_schema(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detects the raw Kaggle fake-reviews schema vs our pipeline
    schema. Raises ValueError if the columns match neither."""
    raw_df = raw_df.copy()
    raw_df.columns = [c.strip() for c in raw_df.columns]
    if KAGGLE_COLS.issubset(raw_df.columns):
        return adapt_kaggle_dataframe(raw_df)
    if OUR_COLS.issubset(raw_df.columns):
        return raw_df
    raise ValueError(
        f"CSV columns don't match either supported schema.\n\n"
        f"Found: {list(raw_df.columns)}\n\n"
        f"Expected either our schema {sorted(OUR_COLS)} or the raw Kaggle schema {sorted(KAGGLE_COLS)}."
    )


def downsample(df: pd.DataFrame, max_rows: int = None, seed: int = 42) -> pd.DataFrame:
    """Stratified on ground_truth_label when present, random otherwise."""
    if not max_rows or len(df) <= max_rows:
        return df
    if "ground_truth_label" in df.columns:
        return _stratified_sample(df, "ground_truth_label", max_rows, seed=seed)
    return df.sample(max_rows, random_state=seed).reset_index(drop=True)


def load_raw(source: str, uploaded_df: pd.DataFrame = None, max_rows: int = None) -> pd.DataFrame:
    """source: a key of DATA_SOURCES, or "upload" together with uploaded_df."""
    if source == "upload":
        if uploaded_df is None:
            raise ValueError("source='upload' needs uploaded_df")
        df = adapt_schema(uploaded_df)
    elif source == "sample":
        df = load_reviews(DATA_SOURCES["sample"])
    elif source in DATA_SOURCES:
        df = adapt_schema(pd.read_csv(DATA_SOURCES[source]))
    else:
        raise ValueError(f"Unknown data source: {source}")
    return downsample(df, max_rows)


@dataclass
class PipelineResult:
    df: pd.DataFrame          # preprocessed + features
    clf: object               # trained RandomForest
    scored: pd.DataFrame      # + credibility_score / label / confidence
    explained: pd.DataFrame   # + positive/negative factors, explanation_text
    train_idx: np.ndarray     # positional indices into df (shared split)
    test_idx: np.ndarray


def run_pipeline(raw_df: pd.DataFrame) -> PipelineResult:
    df = preprocess(raw_df)
    df = extract_features(df)
    clf, _, _ = train_model(df)  # trained fresh on this dataset each run (demo simplicity)
    _, train_idx, test_idx = labeled_split(df)
    scored = score_reviews(df, clf)
    explained = explain_dataframe(scored)
    return PipelineResult(df, clf, scored, explained, train_idx, test_idx)
