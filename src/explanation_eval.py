"""
explanation_eval.py
-------------------
Evaluates the explanations instead of just displaying them.

- Agreement: do the z-score, SHAP and LIME methods pick the same
  features as most important for a review?
- Faithfulness (deletion test): reset a review's top-k "most important"
  features to the dataset mean and measure how much the predicted
  probability moves. A faithful method should move it more than
  resetting k random features.
- Class contrast: average SHAP contribution of each feature for genuine
  vs fake reviews -- what makes the model call something fake?
"""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from features import FEATURE_COLUMNS

METHODS = ["zscore", "shap", "lime"]


def ranking_from_scores(scores: dict) -> list:
    """Features ordered by |score|, most important first."""
    return sorted(FEATURE_COLUMNS, key=lambda f: -abs(scores.get(f, 0.0)))


def zscore_scores(row: pd.Series, means: pd.Series, stds: pd.Series) -> dict:
    return {f: (row[f] - means[f]) / (stds[f] or 1e-6) for f in FEATURE_COLUMNS}


def method_agreement(rankings: dict) -> dict:
    """rankings: {method: [ranking per review, ...]} (same reviews, same order).
    Returns top-1 match rate, mean top-3 Jaccard and mean Kendall tau per pair."""
    out = {"top1_match": {}, "top3_jaccard": {}, "kendall_tau": {}}
    methods = [m for m in METHODS if m in rankings]
    for a, b in itertools.combinations(methods, 2):
        key = f"{a}_vs_{b}"
        top1, jacc, taus = [], [], []
        for ra, rb in zip(rankings[a], rankings[b]):
            top1.append(ra[0] == rb[0])
            sa, sb = set(ra[:3]), set(rb[:3])
            jacc.append(len(sa & sb) / len(sa | sb))
            pos_a = [ra.index(f) for f in FEATURE_COLUMNS]
            pos_b = [rb.index(f) for f in FEATURE_COLUMNS]
            tau, _ = kendalltau(pos_a, pos_b)
            taus.append(0.0 if np.isnan(tau) else tau)
        out["top1_match"][key] = round(float(np.mean(top1)), 4)
        out["top3_jaccard"][key] = round(float(np.mean(jacc)), 4)
        out["kendall_tau"][key] = round(float(np.mean(taus)), 4)
    return out


def deletion_faithfulness(clf, X: pd.DataFrame, rankings: dict, means: pd.Series,
                          ks=(1, 2, 3), seed: int = 42) -> dict:
    """Mean |change in P(genuine)| when each review's top-k features
    (per method) are replaced by the dataset mean; 'random' = k random
    features, the baseline every method should beat."""
    rng = np.random.default_rng(seed)
    X = X[FEATURE_COLUMNS].astype(float).reset_index(drop=True)
    credible = list(clf.classes_).index(1)
    base = clf.predict_proba(X)[:, credible]

    def drop_for(choose):
        drops = []
        for k in ks:
            X_mod = X.copy()
            for i in range(len(X)):
                cols = choose(i, k)
                X_mod.loc[i, cols] = means[cols].to_numpy()
            after = clf.predict_proba(X_mod)[:, credible]
            drops.append(round(float(np.abs(base - after).mean()), 4))
        return drops

    out = {m: drop_for(lambda i, k, m=m: rankings[m][i][:k]) for m in rankings}
    random_order = [list(rng.permutation(FEATURE_COLUMNS)) for _ in range(len(X))]
    out["random"] = drop_for(lambda i, k: random_order[i][:k])
    return {"k": list(ks), "prob_change": out}


def class_contrast(shap_vals: np.ndarray, labels: pd.Series) -> dict:
    """Mean SHAP contribution per feature for genuine (1) and fake (0) reviews."""
    labels = np.asarray(labels)
    out = {}
    for value, name in [(1, "genuine"), (0, "fake")]:
        mask = labels == value
        if mask.any():
            out[name] = {f: round(float(v), 5) for f, v in zip(FEATURE_COLUMNS, shap_vals[mask].mean(axis=0))}
    return out
