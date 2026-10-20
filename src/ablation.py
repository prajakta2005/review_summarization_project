"""
ablation.py
-----------
Ablation study: retrain without (or with only) some of the credibility
features and measure how much performance changes. This is direct
evidence of what each feature contributes, which impurity-based
feature_importances_ can't give (it is biased toward features with many
distinct values, like `length`).

Every score is k-fold stratified cross-validation (mean ± std) using the
same model pipelines as the model comparison (model_zoo).
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from features import FEATURE_COLUMNS
from model_zoo import build_pipeline
from splits import SEED

ABLATION_MODELS = {"rf": "rf_handcrafted", "lr": "lr_handcrafted"}
SCORING = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}


def cv_score(df: pd.DataFrame, y: pd.Series, model_id: str, columns: list, k: int = 5) -> dict:
    """{metric: {"mean", "std", "folds"}} for the model trained on `columns` only."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    scores = cross_validate(build_pipeline(model_id, columns=list(columns)), df, y,
                            cv=cv, scoring=SCORING, n_jobs=-1)
    return {m: {"mean": round(float(scores[f"test_{m}"].mean()), 4),
                "std": round(float(scores[f"test_{m}"].std()), 4),
                "folds": [round(float(v), 4) for v in scores[f"test_{m}"]]}
            for m in SCORING}


def leave_one_out(df, y, model_id, full: dict, k: int = 5) -> list:
    rows = []
    for feature in FEATURE_COLUMNS:
        kept = [f for f in FEATURE_COLUMNS if f != feature]
        score = cv_score(df, y, model_id, kept, k)
        rows.append({"removed": feature, **score,
                     "delta_f1": round(score["f1"]["mean"] - full["f1"]["mean"], 4),
                     "delta_roc_auc": round(score["roc_auc"]["mean"] - full["roc_auc"]["mean"], 4)})
    return sorted(rows, key=lambda r: r["delta_f1"])


def single_feature(df, y, model_id, k: int = 5) -> list:
    rows = [{"feature": f, **cv_score(df, y, model_id, [f], k)} for f in FEATURE_COLUMNS]
    return sorted(rows, key=lambda r: -r["f1"]["mean"])


def forward_selection(df, y, model_id, k: int = 5) -> list:
    """Greedy: at each step add the feature that gives the best CV F1."""
    selected, remaining, steps = [], list(FEATURE_COLUMNS), []
    while remaining:
        candidates = {f: cv_score(df, y, model_id, selected + [f], k) for f in remaining}
        best = max(candidates, key=lambda f: candidates[f]["f1"]["mean"])
        selected.append(best)
        remaining.remove(best)
        steps.append({"step": len(selected), "added": best, "features": list(selected), **candidates[best]})
    return steps
