"""
preprocessing.py
-----------------
Cleans and normalizes review text WHILE preserving raw-text statistics
(punctuation density, repetition, casing) because these are actually
credibility signals, not noise, for this system.

Teaching note: in most NLP pipelines you'd strip punctuation/casing
aggressively before feature extraction. Here we deliberately keep a
"raw" and a "clean" version of each review side by side, because the
Feature Extraction module needs both.
"""

import re
import pandas as pd


def load_reviews(csv_path: str) -> pd.DataFrame:
    """Load reviews from CSV. Expects columns:
    review_id, product_id, reviewer_id, rating, timestamp, review_text
    """
    df = pd.read_csv(csv_path)
    required_cols = {"review_id", "product_id", "reviewer_id", "rating", "timestamp", "review_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate review text (common fake-review pattern:
    the same text posted under multiple reviewer_ids)."""
    before = len(df)
    df = df.drop_duplicates(subset=["review_text"]).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"[preprocessing] Removed {removed} duplicate review(s).")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no review text; fill missing rating with median."""
    df = df.dropna(subset=["review_text"]).reset_index(drop=True)
    if df["rating"].isna().any():
        median_rating = df["rating"].median()
        df["rating"] = df["rating"].fillna(median_rating)
    return df


def clean_text(text: str) -> str:
    """Standard normalization for embedding/model input:
    lowercase, strip URLs/extra whitespace, remove non-alphanumeric noise
    (but keep sentence-ending punctuation for sentence tokenization later).
    """
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"\s+", " ", text)                      # collapse whitespace
    text = text.strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline. Adds a `clean_text` column while
    keeping the original `review_text` (raw) intact for feature extraction.
    """
    df = drop_duplicates(df)
    df = handle_missing(df)
    df["clean_text"] = df["review_text"].apply(clean_text)
    df["clean_text_lower"] = df["clean_text"].str.lower()
    return df


if __name__ == "__main__":
    df = load_reviews("data/sample_reviews.csv")
    df = preprocess(df)
    print(df[["review_id", "review_text", "clean_text"]].head())
