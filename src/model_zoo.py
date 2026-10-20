"""
model_zoo.py
------------
Every classical model in the comparison, as a factory returning an
sklearn Pipeline (feature transformer + classifier) that is fit on the
featurized review DataFrame. DistilBERT is trained separately on Colab
(notebooks/bert_finetune_colab.ipynb) and merged in from its predictions.

rf_handcrafted is exactly the deployed credibility model
(credibility_model.train_model) so the comparison includes what the
dashboard actually runs.
"""

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from text_features import build_feature_transformer, build_tfidf, TEXT_COLUMN


@dataclass
class ModelSpec:
    model_id: str
    model: str            # family name shown in tables
    feature_set: str      # handcrafted | tfidf | hybrid | bert
    make_classifier: Callable
    scale: bool = True    # standardize handcrafted features (not needed for trees)
    interpretable: bool = False


MODEL_SPECS = {s.model_id: s for s in [
    ModelSpec("rf_handcrafted", "Random Forest", "handcrafted",
              lambda: RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42,
                                             class_weight="balanced"),
              scale=False, interpretable=True),
    ModelSpec("rf_handcrafted_tuned", "Random Forest (tuned)", "handcrafted",
              lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42,
                                             class_weight="balanced", n_jobs=-1),
              scale=False, interpretable=True),
    ModelSpec("lr_handcrafted", "Logistic Regression", "handcrafted",
              lambda: LogisticRegression(max_iter=1000), interpretable=True),
    ModelSpec("svm_handcrafted", "SVM (RBF)", "handcrafted",
              lambda: SVC(kernel="rbf", random_state=42), interpretable=True),
    ModelSpec("rf_tfidf", "Random Forest", "tfidf",
              lambda: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ModelSpec("lr_tfidf", "Logistic Regression", "tfidf",
              lambda: LogisticRegression(C=4, max_iter=2000)),
    ModelSpec("svm_tfidf", "SVM (linear)", "tfidf",
              lambda: LinearSVC(C=0.5, max_iter=5000, random_state=42)),
    ModelSpec("lr_hybrid", "Logistic Regression", "hybrid",
              lambda: LogisticRegression(C=4, max_iter=2000)),
    ModelSpec("svm_hybrid", "SVM (linear)", "hybrid",
              lambda: LinearSVC(C=0.5, max_iter=5000, random_state=42)),
]}

BERT_SPEC = ModelSpec("distilbert", "DistilBERT (fine-tuned)", "bert", make_classifier=None)


def build_pipeline(model_id: str, columns: list = None) -> Pipeline:
    """columns: optional subset of handcrafted features (ablation study)."""
    spec = MODEL_SPECS[model_id]
    return Pipeline([
        ("features", build_feature_transformer(spec.feature_set, scale=spec.scale, columns=columns)),
        ("clf", spec.make_classifier()),
    ])


def positive_score(model, X) -> np.ndarray:
    """Probability of class 1 (genuine) when available, otherwise the
    decision-function margin (SVMs). Both work for ROC/PR curves."""
    if hasattr(model, "predict_proba"):
        try:
            classes = list(model.classes_)
            return model.predict_proba(X)[:, classes.index(1)]
        except AttributeError:
            pass
    return model.decision_function(X)


def fit_and_predict(model_id: str, df: pd.DataFrame, y: pd.Series,
                    train_idx: np.ndarray, test_idx: np.ndarray, columns: list = None) -> dict:
    model = build_pipeline(model_id, columns=columns)
    X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
    y_train = y.iloc[train_idx]

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    score = positive_score(model, X_test)
    predict_seconds = time.perf_counter() - start

    return {
        "model": model,
        "y_true": y.iloc[test_idx].to_numpy(),
        "y_pred": np.asarray(y_pred),
        "score": np.asarray(score),
        "train_seconds": round(train_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
    }


def build_text_explainer_model() -> Pipeline:
    """TF-IDF + Logistic Regression on raw strings (not a DataFrame), the
    form LIME's text explainer needs: model.predict_proba(list_of_texts)."""
    return Pipeline([("tfidf", build_tfidf()), ("clf", LogisticRegression(C=4, max_iter=2000))])


__all__ = ["MODEL_SPECS", "BERT_SPEC", "build_pipeline", "fit_and_predict",
           "positive_score", "build_text_explainer_model", "TEXT_COLUMN"]
