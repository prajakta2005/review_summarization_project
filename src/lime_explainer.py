"""
lime_explainer.py
-----------------
LIME explanations: fit a simple local linear model around ONE review by
perturbing it and watching how the prediction changes.

- Tabular LIME explains the deployed Random Forest on the 6 features
  (a second opinion next to SHAP).
- Text LIME explains the TF-IDF + Logistic Regression text model
  (models/tfidf_lr.joblib from the model comparison) and shows WHICH
  WORDS pushed a review toward genuine or fake.

Positive weight = pushes toward genuine (class 1), negative = toward fake.
"""

import html
import os
import re

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer

from features import FEATURE_COLUMNS

TEXT_MODEL_PATH = "models/tfidf_lr.joblib"
CLASS_NAMES = ["fake", "genuine"]
SEED = 42


def lime_tabular_explain(clf, X_train: pd.DataFrame, x_row: pd.Series,
                         num_features: int = 6, num_samples: int = 2000) -> list:
    """[{"feature", "condition", "weight"}, ...] sorted by |weight|."""
    explainer = LimeTabularExplainer(
        X_train[FEATURE_COLUMNS].to_numpy(), feature_names=FEATURE_COLUMNS,
        class_names=CLASS_NAMES, mode="classification",
        discretize_continuous=True, random_state=SEED,
    )

    def predict_fn(arr):
        return clf.predict_proba(pd.DataFrame(arr, columns=FEATURE_COLUMNS))

    exp = explainer.explain_instance(
        x_row[FEATURE_COLUMNS].to_numpy(dtype=float), predict_fn,
        num_features=num_features, labels=(1,), num_samples=num_samples,
    )
    conditions = dict(exp.as_list(label=1))
    out = []
    for idx, weight in exp.as_map()[1]:
        # as_list and as_map share order, so match the condition text by position
        condition = list(conditions)[len(out)]
        out.append({"feature": FEATURE_COLUMNS[idx], "condition": condition, "weight": float(weight)})
    return sorted(out, key=lambda d: -abs(d["weight"]))


def load_text_model(train_texts=None, train_labels=None):
    """The saved TF-IDF + LR text model, or a quick one trained on the
    given texts (about 5 s on the full training split) if it's missing."""
    if os.path.exists(TEXT_MODEL_PATH):
        return joblib.load(TEXT_MODEL_PATH)
    if train_texts is None:
        raise FileNotFoundError(
            f"{TEXT_MODEL_PATH} not found -- run experiments/run_model_comparison.py, "
            "or pass training texts to train a fallback model.")
    from model_zoo import build_text_explainer_model
    return build_text_explainer_model().fit(train_texts, train_labels)


def lime_text_explain(text_model, text: str, num_features: int = 10, num_samples: int = 1000) -> list:
    """[(word, weight), ...] sorted by |weight|."""
    explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=SEED)
    exp = explainer.explain_instance(text, text_model.predict_proba, num_features=num_features,
                                     labels=(1,), num_samples=num_samples)
    return sorted(((str(w), float(v)) for w, v in exp.as_list(label=1)), key=lambda t: -abs(t[1]))


def highlight_html(text: str, word_weights: list) -> str:
    """The review as HTML with explained words highlighted: green pushes
    toward genuine, red toward fake, stronger colour = bigger weight.
    Text is HTML-escaped, so it is safe for unsafe_allow_html."""
    weights = dict(word_weights)
    biggest = max((abs(w) for w in weights.values()), default=0) or 1
    parts = []
    # LIME's text explainer splits on non-word characters, so do the same
    for token in re.split(r"(\W+)", text):
        if token in weights:
            w = weights[token]
            alpha = 0.15 + 0.6 * abs(w) / biggest
            color = f"rgba(46, 204, 113, {alpha:.2f})" if w > 0 else f"rgba(231, 76, 60, {alpha:.2f})"
            parts.append(f'<span style="background:{color};border-radius:3px;padding:0 2px" '
                         f'title="{w:+.3f}">{html.escape(token)}</span>')
        else:
            parts.append(html.escape(token))
    return '<div style="line-height:1.8">' + "".join(parts) + "</div>"
