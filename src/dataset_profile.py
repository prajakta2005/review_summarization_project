"""
dataset_profile.py
------------------
Describes a (preprocessed + featurized) review dataset: size, class
balance, categories, ratings, text length and how each credibility
feature is distributed for genuine vs fake reviews.

Output is plain JSON-serializable dicts (histograms are pre-binned) so
the full-dataset profile can be saved to results/dataset_profile.json
and drawn without shipping 40k raw rows to the browser.
"""

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS

LABEL_NAMES = {1: "genuine", 0: "fake"}
N_BINS = 30


def label_groups(df: pd.DataFrame) -> pd.Series:
    """'genuine' / 'fake' from ground_truth_label, or 'unlabeled'."""
    if "ground_truth_label" in df.columns and df["ground_truth_label"].notna().all():
        return df["ground_truth_label"].astype(int).map(LABEL_NAMES)
    return pd.Series("unlabeled", index=df.index)


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Standardized mean difference -- how well one feature separates
    two groups (|d| ~0.2 small, ~0.5 medium, ~0.8 large)."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled else 0.0


def _hist_by_group(values: pd.Series, groups: pd.Series) -> dict:
    # clip the long tail (e.g. a few 1,000-word reviews) so bins stay readable
    upper = values.quantile(0.99)
    clipped = values.clip(upper=upper)
    edges = np.histogram_bin_edges(clipped, bins=N_BINS)
    out = {"bins": [round(float(e), 4) for e in edges]}
    for name in sorted(groups.unique()):
        counts, _ = np.histogram(clipped[groups == name], bins=edges)
        out[name] = counts.tolist()
    return out


def profile_dataset(df: pd.DataFrame) -> dict:
    groups = label_groups(df)
    group_names = sorted(groups.unique())

    category_col = "category" if "category" in df.columns else "product_id"
    categories = (
        pd.crosstab(df[category_col], groups)
        .reindex(columns=group_names, fill_value=0)
        .assign(total=lambda t: t.sum(axis=1))
        .sort_values("total", ascending=False)
        .reset_index()
        .rename(columns={category_col: "category"})
    )

    ratings = pd.crosstab(df["rating"].astype(int), groups).reindex(columns=group_names, fill_value=0)

    feature_stats = {}
    for col in FEATURE_COLUMNS:
        stats = {g: {"mean": round(float(df.loc[groups == g, col].mean()), 4),
                     "std": round(float(df.loc[groups == g, col].std(ddof=1) or 0), 4)}
                 for g in group_names}
        if {"genuine", "fake"} <= set(group_names):
            stats["cohens_d"] = round(cohens_d(df.loc[groups == "genuine", col],
                                               df.loc[groups == "fake", col]), 4)
        feature_stats[col] = stats

    length_by_label = {
        g: {"mean": round(float(df.loc[groups == g, "length"].mean()), 1),
            "median": float(df.loc[groups == g, "length"].median()),
            "p95": float(df.loc[groups == g, "length"].quantile(0.95))}
        for g in group_names
    }

    return {
        "n_raw": int(df.attrs.get("n_raw", len(df))),
        "n_after_preprocess": int(len(df)),
        "duplicates_removed": int(df.attrs.get("duplicates_removed", 0)),
        "labeled": "unlabeled" not in group_names,
        "label_counts": {g: int((groups == g).sum()) for g in group_names},
        "n_categories": int(df[category_col].nunique()),
        "categories": categories.to_dict(orient="records"),
        "rating_by_label": {g: {str(r): int(c) for r, c in ratings[g].items()} for g in group_names},
        "length_by_label": length_by_label,
        "feature_stats_by_label": feature_stats,
        "feature_hist_by_label": {col: _hist_by_group(df[col], groups) for col in FEATURE_COLUMNS},
    }
