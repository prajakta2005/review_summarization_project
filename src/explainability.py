"""
explainability.py
------------------
Turns model internals into human-readable reasons.

Two levels, matching what the patent describes:
1. Global importance: which features matter most across ALL reviews
   (straight from the RandomForest's feature_importances_).
2. Per-review explanation: for THIS review, which of its own feature
   values are unusually high/low compared to the training data, and
   what that implies in plain English.

Design choice: rather than SHAP (which needs extra dependencies and
is overkill for a 6-feature interpretable model), we use the
statistically simple but transparent approach of comparing each
review's feature values to the dataset mean/std. This keeps the
"why" fully auditable -- you can show a reviewer/examiner the exact
arithmetic behind every explanation. SHAP is a natural upgrade path
once you have a larger real dataset (mention this in your report).
"""

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS

# Human-readable meaning of "this feature being HIGH" vs "LOW"
FEATURE_MEANING = {
    "sentiment": {
        "high": "strongly positive language",
        "low": "strongly negative language",
    },
    "length": {
        "high": "a detailed, longer review",
        "low": "an unusually short review",
    },
    "exclaim_density": {
        "high": "excessive exclamation marks (hype-style language)",
        "low": "measured, non-hype punctuation",
    },
    "repetition_ratio": {
        "high": "repetitive wording",
        "low": "varied, non-repetitive wording",
    },
    "specificity": {
        "high": "specific, concrete details (numbers, timeframes)",
        "low": "a lack of specific, concrete details",
    },
    "rating_consistency": {
        "high": "a star rating that matches the tone of the text",
        "low": "a star rating that conflicts with the tone of the text",
    },
}

# Direction that HELPS credibility for each feature (True = high is good)
HIGH_IS_CREDIBLE = {
    "sentiment": None,           # neutral -- sentiment itself isn't a credibility signal
    "length": True,
    "exclaim_density": False,
    "repetition_ratio": False,
    "specificity": True,
    "rating_consistency": True,
}


def global_feature_importance(clf) -> pd.Series:
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS)
    return importances.sort_values(ascending=False)


def explain_review(row: pd.Series, dataset_stats: pd.DataFrame, top_k: int = 3) -> dict:
    """dataset_stats: df.describe() style stats (mean/std) over FEATURE_COLUMNS,
    used to judge whether THIS review's value is unusually high/low.
    Returns a dict with positive_factors, negative_factors, and a natural
    language explanation string.
    """
    z_scores = {}
    for col in FEATURE_COLUMNS:
        mean = dataset_stats.loc["mean", col]
        std = dataset_stats.loc["std", col] or 1e-6
        z_scores[col] = (row[col] - mean) / std

    positive_factors, negative_factors = [], []

    for col, z in z_scores.items():
        direction = "high" if z > 0 else "low"
        credible_dir = HIGH_IS_CREDIBLE.get(col)
        if credible_dir is None:
            continue  # sentiment alone is not judged as good/bad for credibility
        is_credible_signal = (direction == "high") == credible_dir
        strength = abs(z)
        entry = (col, strength, FEATURE_MEANING[col][direction])
        if is_credible_signal:
            positive_factors.append(entry)
        else:
            negative_factors.append(entry)

    positive_factors.sort(key=lambda x: -x[1])
    negative_factors.sort(key=lambda x: -x[1])

    top_pos = positive_factors[:top_k]
    top_neg = negative_factors[:top_k]

    explanation_parts = []
    if top_pos:
        phrases = [p[2] for p in top_pos]
        explanation_parts.append(
            "The review shows " + ", ".join(phrases) + ", which support(s) credibility."
        )
    if top_neg:
        phrases = [p[2] for p in top_neg]
        explanation_parts.append(
            "However, it also shows " + ", ".join(phrases) + ", which raise(s) concern."
        )
    if not explanation_parts:
        explanation_parts.append("No strongly distinguishing signals were found for this review.")

    return {
        "positive_factors": [p[2] for p in top_pos],
        "negative_factors": [n[2] for n in top_neg],
        "explanation_text": " ".join(explanation_parts),
    }


def explain_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Adds explanation columns to every row of the scored dataframe."""
    stats = df[FEATURE_COLUMNS].describe()
    records = df.apply(lambda row: explain_review(row, stats), axis=1)
    df = df.copy()
    df["positive_factors"] = records.apply(lambda r: r["positive_factors"])
    df["negative_factors"] = records.apply(lambda r: r["negative_factors"])
    df["explanation_text"] = records.apply(lambda r: r["explanation_text"])
    return df


if __name__ == "__main__":
    from preprocessing import load_reviews, preprocess
    from features import extract_features
    from credibility_model import train_model, score_reviews

    df = load_reviews("data/sample_reviews.csv")
    df = preprocess(df)
    df = extract_features(df)
    clf, _, _ = train_model(df)

    print("Global feature importance:")
    print(global_feature_importance(clf))
    print()

    scored = score_reviews(df, clf)
    explained = explain_dataframe(scored)

    for _, row in explained.iterrows():
        print(f"\nReview {row['review_id']} | Credibility: {row['credibility_score']}% ({row['credibility_label']})")
        print(f"  Text: {row['review_text'][:80]}...")
        print(f"  Explanation: {row['explanation_text']}")
