"""
evaluation.py
-------------
Module 12: Performance Evaluation. Accuracy, precision, recall, F1,
ROC-AUC, confusion matrix -- flags whether labels were real or heuristic.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

from features import FEATURE_COLUMNS
from credibility_model import build_training_labels


def evaluate_model(clf, df: pd.DataFrame) -> dict:
    df = build_training_labels(df)
    label_source = ("real_ground_truth" if "ground_truth_label" in df.columns
                     and df["ground_truth_label"].notna().all() else "heuristic_pseudo_labels")

    X = df[FEATURE_COLUMNS]
    y = df["label"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )

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