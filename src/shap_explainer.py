"""
shap_explainer.py
-----------------
SHAP explanations for the deployed credibility model (the Random Forest
on the 6 handcrafted features).

TreeExplainer computes EXACT Shapley values for tree ensembles: for each
review, the base value (average predicted credibility) plus the 6
feature contributions adds up to the model's predicted probability.
Positive contribution = pushes toward credible/genuine (class 1),
negative = pushes toward fake.
"""

import numpy as np
import pandas as pd
import shap

from features import FEATURE_COLUMNS

_EXPLAINERS = {}


def get_tree_explainer(clf) -> shap.TreeExplainer:
    """One explainer per fitted model (building it walks every tree)."""
    key = id(clf)
    if key not in _EXPLAINERS:
        _EXPLAINERS[key] = shap.TreeExplainer(clf)
    return _EXPLAINERS[key]


def _credible_index(clf) -> int:
    return list(clf.classes_).index(1)


def base_value(clf) -> float:
    expected = np.atleast_1d(get_tree_explainer(clf).expected_value)
    return float(expected[_credible_index(clf)] if len(expected) > 1 else expected[0])


def shap_values(clf, X: pd.DataFrame) -> np.ndarray:
    """(n_reviews, 6) contributions toward the credible class."""
    values = get_tree_explainer(clf).shap_values(X[FEATURE_COLUMNS])
    if isinstance(values, list):          # older shap: one array per class
        return np.asarray(values[_credible_index(clf)])
    values = np.asarray(values)
    if values.ndim == 3:                  # newer shap: (n, features, classes)
        return values[:, :, _credible_index(clf)]
    return values


def global_shap_importance(clf, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature, i.e. average size of each feature's push."""
    vals = shap_values(clf, X)
    return pd.Series(np.abs(vals).mean(axis=0), index=FEATURE_COLUMNS).sort_values(ascending=False)


def shap_row_factors(shap_row, x_row) -> list:
    """[(feature, feature_value, contribution), ...] sorted by |contribution|."""
    factors = [(f, float(x_row[f]), float(c)) for f, c in zip(FEATURE_COLUMNS, shap_row)]
    return sorted(factors, key=lambda t: -abs(t[2]))
