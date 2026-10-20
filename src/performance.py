"""
performance.py
--------------
Deeper performance analysis on top of evaluation.compute_metrics:
per-class report, per-category metrics, decision-threshold sweep and
McNemar's significance test between two models' test-set predictions.

Works on the saved test predictions in results/predictions/<model>.csv
(review_id, y_true, y_pred, score), so nothing needs retraining.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score

PREDICTIONS_DIR = "results/predictions"
CLASS_NAMES = {0: "fake", 1: "genuine"}


def load_predictions(model_id: str) -> pd.DataFrame:
    path = os.path.join(PREDICTIONS_DIR, f"{model_id}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def available_prediction_models() -> list:
    if not os.path.isdir(PREDICTIONS_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(PREDICTIONS_DIR) if f.endswith(".csv"))


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": support},
                        index=[CLASS_NAMES[0], CLASS_NAMES[1]]).round(4)


def per_category_metrics(pred_df: pd.DataFrame, categories: pd.Series) -> pd.DataFrame:
    """categories: Series mapping review_id -> category."""
    df = pred_df.assign(category=pred_df["review_id"].map(categories))
    rows = []
    for cat, g in df.groupby("category"):
        rows.append({"category": cat, "n": len(g),
                     "accuracy": accuracy_score(g["y_true"], g["y_pred"]),
                     "f1": f1_score(g["y_true"], g["y_pred"], zero_division=0)})
    return pd.DataFrame(rows).sort_values("f1").round(4)


def threshold_sweep(y_true, score, steps: int = 101) -> pd.DataFrame:
    """Metrics at each decision threshold. Probabilities sweep 0..1;
    margins (SVM decision_function) sweep their own quantiles."""
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    is_probability = score.min() >= 0 and score.max() <= 1
    thresholds = (np.linspace(0, 1, steps) if is_probability
                  else np.unique(np.quantile(score, np.linspace(0, 1, steps))))
    rows = []
    for t in thresholds:
        pred = (score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append({
            "threshold": float(t), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "accuracy": (tp + tn) / len(y_true), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })
    return pd.DataFrame(rows)


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """McNemar's test on paired predictions over the same test rows.
    b = A right & B wrong, c = A wrong & B right. Exact binomial test when
    b + c < 25, otherwise chi-squared with continuity correction."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_right, b_right = pred_a == y_true, pred_b == y_true
    b = int((a_right & ~b_right).sum())
    c = int((~a_right & b_right).sum())
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "no disagreements"}
    if n < 25:
        p = binomtest(b, n, 0.5).pvalue
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": float(p), "method": "exact binomial"}
    stat = (abs(b - c) - 1) ** 2 / n
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(chi2.sf(stat, 1)),
            "method": "chi-squared (continuity corrected)"}
