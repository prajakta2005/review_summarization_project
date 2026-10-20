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

Two per-review methods, same output format:
- "zscore" (default): compares each review's feature values to the
  dataset mean/std. Fully auditable arithmetic, but it doesn't look at
  the model -- whether a high value is "good" comes from the fixed
  HIGH_IS_CREDIBLE table below.
- "shap": uses the model's exact SHAP contributions (shap_explainer.py),
  so whether a feature counted for or against credibility is what the
  model actually did for THIS review. The z-score table assumes
  exclamation marks signal fakes, but on the full dataset genuine
  reviews use slightly more of them; SHAP follows the model instead.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

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


def permutation_feature_importance(clf, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5) -> pd.Series:
    """How much the score (F1) drops when each feature's values are shuffled.
    Unlike feature_importances_ (impurity), this isn't biased toward
    features with many distinct values. Use held-out rows for X, y."""
    result = permutation_importance(clf, X[FEATURE_COLUMNS], y, scoring="f1",
                                    n_repeats=n_repeats, random_state=42, n_jobs=-1)
    return pd.Series(result.importances_mean, index=FEATURE_COLUMNS).sort_values(ascending=False)


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


MIN_SHAP_CONTRIBUTION = 0.005  # ignore pushes smaller than 0.5 percentage points


def explain_review_shap(row: pd.Series, shap_row, feature_means: pd.Series, top_k: int = 3) -> dict:
    """Same output as explain_review, but driven by SHAP: a feature is a
    positive factor if it pushed THIS review's credibility up, negative
    if it pushed it down. The phrase still says whether the value is
    high or low relative to the dataset mean."""
    positive, negative = [], []
    for col, contribution in zip(FEATURE_COLUMNS, shap_row):
        if abs(contribution) < MIN_SHAP_CONTRIBUTION:
            continue
        direction = "high" if row[col] > feature_means[col] else "low"
        entry = (col, abs(contribution), FEATURE_MEANING[col][direction])
        (positive if contribution > 0 else negative).append(entry)

    positive.sort(key=lambda x: -x[1])
    negative.sort(key=lambda x: -x[1])
    top_pos, top_neg = positive[:top_k], negative[:top_k]

    parts = []
    if top_pos:
        parts.append("The model rated this review more credible because of "
                     + ", ".join(p[2] for p in top_pos) + ".")
    if top_neg:
        parts.append(("However, " if top_pos else "The model rated it less credible because of ")
                     + ", ".join(n[2] for n in top_neg)
                     + (" lowered its credibility." if top_pos else "."))
    if not parts:
        parts.append("No single signal moved this review's score much; it sits near the average.")
    return {
        "positive_factors": [p[2] for p in top_pos],
        "negative_factors": [n[2] for n in top_neg],
        "explanation_text": " ".join(parts),
    }


def explain_dataframe(df: pd.DataFrame, method: str = "zscore", clf=None) -> pd.DataFrame:
    """Adds explanation columns to every row of the scored dataframe.
    method="shap" needs the fitted credibility model as clf."""
    if method == "shap":
        if clf is None:
            raise ValueError("method='shap' needs clf")
        from shap_explainer import shap_values
        vals = shap_values(clf, df)  # one vectorized call for all rows
        means = df[FEATURE_COLUMNS].mean()
        records = pd.Series([explain_review_shap(row, vals[i], means)
                             for i, (_, row) in enumerate(df.iterrows())], index=df.index)
    elif method == "zscore":
        stats = df[FEATURE_COLUMNS].describe()
        records = df.apply(lambda row: explain_review(row, stats), axis=1)
    else:
        raise ValueError(f"Unknown explanation method: {method}")
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
