"""
evaluation.py
-------------
Module 12: Performance Evaluation. Accuracy, precision, recall, F1,
ROC-AUC, confusion matrix -- flags whether labels were real or heuristic.

`compute_metrics` is the ONE place metrics are computed, so the model
comparison, cross-validation and dashboard numbers are always comparable.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score, roc_curve, precision_recall_curve
)

from features import FEATURE_COLUMNS
from credibility_model import labeled_split


MAX_CURVE_POINTS = 200


def _downsample_curve(xs, ys, max_points=MAX_CURVE_POINTS):
    if len(xs) <= max_points:
        idx = np.arange(len(xs))
    else:
        idx = np.unique(np.linspace(0, len(xs) - 1, max_points).round().astype(int))
    return [round(float(xs[i]), 4) for i in idx], [round(float(ys[i]), 4) for i in idx]


def compute_metrics(y_true, y_pred, score=None, curves: bool = True) -> dict:
    """Classification metrics with class 1 (= genuine/credible) as positive.
    score: probability of class 1 or any decision score (e.g. SVM margin);
    needed for ROC-AUC, PR-AUC and curves, which are None without it.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "roc_auc": None,
        "pr_auc": None,
    }
    if score is not None and len(np.unique(y_true)) > 1:
        score = np.asarray(score)
        metrics["roc_auc"] = round(roc_auc_score(y_true, score), 4)
        metrics["pr_auc"] = round(average_precision_score(y_true, score), 4)
        if curves:
            fpr, tpr, _ = roc_curve(y_true, score)
            prec, rec, _ = precision_recall_curve(y_true, score)
            metrics["roc_curve"] = dict(zip(["fpr", "tpr"], _downsample_curve(fpr, tpr)))
            metrics["pr_curve"] = dict(zip(["recall", "precision"], _downsample_curve(rec, prec)))
    return metrics


def evaluate_model(clf, df: pd.DataFrame) -> dict:
    df, _, test_idx = labeled_split(df)
    label_source = ("real_ground_truth" if "ground_truth_label" in df.columns
                     and df["ground_truth_label"].notna().all() else "heuristic_pseudo_labels")

    X_test = df[FEATURE_COLUMNS].iloc[test_idx]
    y_test = df["label"].iloc[test_idx]

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]

    metrics = {
        "label_source": label_source,
        "n_test_samples": len(X_test),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    if y_test.nunique() > 1:
        metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)
    else:
        metrics["roc_auc"] = None
    return metrics