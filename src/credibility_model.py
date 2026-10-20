"""
credibility_model.py
---------------------
Trains and serves the credibility classifier.

IMPORTANT (read this before using in production):
There is no real labeled "genuine vs fake" dataset here. Fabricating
one would give a false sense of accuracy. Instead we bootstrap with
HEURISTIC PSEUDO-LABELS built from the same features the model will
learn on -- this is a legitimate, commonly used starting strategy
(weak supervision), but it means the model is currently learning to
approximate a rule, not real-world fraud patterns.

Before you rely on this for your evaluation section (accuracy,
precision/recall, ROC-AUC etc.), swap `pseudo_label()` for real
labels from a public fake-review dataset (e.g. Amazon/Yelp research
datasets) or hand-labeled data from your own team. The rest of the
pipeline (features -> model -> explainability) does not need to
change when you do that swap -- only the training labels do.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from features import FEATURE_COLUMNS

MODEL_PATH = "models/credibility_rf.joblib"


def pseudo_label(row) -> int:
    """Heuristic weak-supervision rule -> 1 = credible, 0 = suspicious.
    Used ONLY to bootstrap training. Replace with real labels when available.
    """
    suspicious_score = 0
    if row["exclaim_density"] > 3:
        suspicious_score += 1
    if row["repetition_ratio"] > 0.25:
        suspicious_score += 1
    if row["specificity"] == 0:
        suspicious_score += 1
    if row["rating_consistency"] < 0.5:
        suspicious_score += 1
    if row["length"] < 6:
        suspicious_score += 1
    # 2+ suspicious signals -> label as not credible
    return 0 if suspicious_score >= 2 else 1


def build_training_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Uses REAL ground-truth labels when present (e.g. from the Kaggle
    fake-reviews-dataset adapter, column `ground_truth_label`). Falls back
    to heuristic pseudo-labels only when no real labels are available
    (e.g. the bundled small sample dataset).
    """
    df = df.copy()
    if "ground_truth_label" in df.columns and df["ground_truth_label"].notna().all():
        print("[credibility_model] Using REAL ground-truth labels for training.")
        df["label"] = df["ground_truth_label"].astype(int)
    else:
        print("[credibility_model] No ground-truth labels found -- using heuristic pseudo-labels (see module docstring).")
        df["label"] = df.apply(pseudo_label, axis=1)
    return df


def train_model(df: pd.DataFrame):
    df = build_training_labels(df)
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    # Small dataset -> keep test split small too, just to demonstrate the pipeline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=4, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    joblib.dump(clf, MODEL_PATH)
    return clf, X_test, y_test


def load_model():
    return joblib.load(MODEL_PATH)


def score_reviews(df: pd.DataFrame, clf) -> pd.DataFrame:
    """Adds credibility_score (0-100), credibility_label (High/Medium/Low),
    and confidence to the dataframe.
    """
    df = df.copy()
    X = df[FEATURE_COLUMNS]
    proba = clf.predict_proba(X)
    credible_idx = list(clf.classes_).index(1)
    scores = proba[:, credible_idx]

    df["credibility_score"] = (scores * 100).round(1)
    df["confidence"] = (np.max(proba, axis=1) * 100).round(1)

    def bucket(score):
        if score >= 70:
            return "High"
        elif score >= 40:
            return "Medium"
        return "Low"

    df["credibility_label"] = df["credibility_score"].apply(bucket)
    return df


if __name__ == "__main__":
    from preprocessing import load_reviews, preprocess
    from features import extract_features

    df = load_reviews("data/sample_reviews.csv")
    df = preprocess(df)
    df = extract_features(df)

    clf, X_test, y_test = train_model(df)
    acc = clf.score(X_test, y_test)
    print(f"Holdout accuracy on pseudo-labels: {acc:.2f} (small demo dataset -- not a real benchmark)")

    scored = score_reviews(df, clf)
    print(scored[["review_id", "credibility_score", "credibility_label", "confidence"]])
