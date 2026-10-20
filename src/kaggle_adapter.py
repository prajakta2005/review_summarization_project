"""
kaggle_adapter.py
------------------
Maps the Kaggle "Fake Reviews Dataset" (mexwell/fake-reviews-dataset)
schema onto our pipeline's expected schema, and preserves its real
ground-truth label so credibility_model.py can train on REAL data
instead of heuristic pseudo-labels.
"""

import sys
import pandas as pd


def _stratified_sample(df: pd.DataFrame, label_col: str, n: int, seed: int = 42) -> pd.DataFrame:
    """Manual stratified sampling that reliably preserves all columns
    (pandas' groupby().apply(lambda g: g.sample(...)) silently drops the
    grouping column in some pandas versions -- this avoids that bug).
    """
    n_classes = df[label_col].nunique()
    per_class = max(n // n_classes, 1)
    parts = []
    for _, group in df.groupby(label_col):
        parts.append(group.sample(min(len(group), per_class), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def adapt_kaggle_dataframe(df: pd.DataFrame, sample_n: int = None) -> pd.DataFrame:
    """Core adapter logic, usable directly on an in-memory dataframe
    (e.g. from Streamlit's file_uploader) -- not just from a file path.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if sample_n and len(df) > sample_n:
        df = _stratified_sample(df, "label", sample_n)

    adapted = pd.DataFrame({
        "review_id": range(1, len(df) + 1),
        "product_id": df["category"].astype(str),
        "reviewer_id": ["U" + str(i) for i in range(1, len(df) + 1)],
        "rating": df["rating"],
        "timestamp": pd.NaT,
        "review_text": df["text_"],
        "ground_truth_label": (df["label"] == "OR").astype(int),
    })
    return adapted


def adapt_kaggle_fake_reviews(kaggle_csv_path: str, out_path: str, sample_n: int = 2000):
    """File-based CLI wrapper -- still supported, no longer required
    (the app now auto-detects and adapts on upload too)."""
    df = pd.read_csv(kaggle_csv_path)
    adapted = adapt_kaggle_dataframe(df, sample_n=sample_n)
    adapted.to_csv(out_path, index=False)
    print(f"Wrote {len(adapted)} rows to {out_path}")
    print(adapted["ground_truth_label"].value_counts(normalize=True))
    return adapted


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/kaggle_fake_reviews.csv"
    adapt_kaggle_fake_reviews(kaggle_csv_path=in_path, out_path="data/kaggle_adapted.csv")