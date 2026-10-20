"""
single_review.py
----------------
Scores ONE typed-in review with the deployed pipeline: clean ->
features -> credibility score -> SHAP + z-score explanations, plus where
the review would land in its category's credibility-aware summary.
Used by the Live Demo page.
"""

import numpy as np
import pandas as pd

from preprocessing import clean_text
from features import extract_features, FEATURE_COLUMNS
from credibility_model import score_reviews
from explainability import explain_review, explain_review_shap
from shap_explainer import shap_values, shap_row_factors, base_value


def score_single_review(text: str, rating: float, category: str, clf, reference: pd.DataFrame) -> dict:
    """reference: the loaded, scored dataset (for feature averages and to
    rank this review within its category)."""
    cleaned = clean_text(text)
    row_df = pd.DataFrame([{
        "review_id": -1, "product_id": category, "reviewer_id": "demo", "rating": float(rating),
        "timestamp": None, "review_text": text, "clean_text": cleaned, "clean_text_lower": cleaned.lower(),
    }])
    row_df = score_reviews(extract_features(row_df), clf)
    row = row_df.iloc[0]

    contributions = shap_values(clf, row_df)[0]
    means = reference[FEATURE_COLUMNS].mean()
    shap_expl = explain_review_shap(row, contributions, means)
    z_expl = explain_review(row, reference[FEATURE_COLUMNS].describe())

    in_category = reference[reference["product_id"] == category]["credibility_score"]
    percentile = float((in_category < row["credibility_score"]).mean()) if len(in_category) else None

    comparison = pd.DataFrame({"this review": row[FEATURE_COLUMNS].astype(float), "dataset mean": means})
    if "ground_truth_label" in reference.columns and reference["ground_truth_label"].notna().all():
        comparison["genuine mean"] = reference.loc[reference["ground_truth_label"] == 1, FEATURE_COLUMNS].mean()
        comparison["fake mean"] = reference.loc[reference["ground_truth_label"] == 0, FEATURE_COLUMNS].mean()

    return {
        "row": row,
        "credibility_score": float(row["credibility_score"]),
        "credibility_label": row["credibility_label"],
        "confidence": float(row["confidence"]),
        "features": {f: float(row[f]) for f in FEATURE_COLUMNS},
        "feature_comparison": comparison.round(3),
        "shap_factors": shap_row_factors(contributions, row),
        "base_value": base_value(clf),
        "shap_explanation": shap_expl,
        "zscore_explanation": z_expl,
        "summary_weight": max(row["credibility_score"] / 100, 0.05),
        "category_percentile": percentile,
    }
