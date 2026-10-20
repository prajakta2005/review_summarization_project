"""
5-fold stratified cross-validation for every classical model, so results
don't rest on one lucky train/test split. (DistilBERT is excluded: 5
fine-tuning runs on Colab is too expensive; its single-split numbers are
marked as such on the dashboard.)

    python experiments/run_cross_validation.py           # full dataset, 5 folds (~15 min)
    python experiments/run_cross_validation.py --quick   # 10k rows, 3 folds
    python experiments/run_cross_validation.py --only lr_tfidf,rf_handcrafted

Writes results/cv_results.json.
"""

import argparse
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.model_selection import StratifiedKFold, cross_validate

from pipeline import load_raw
from preprocessing import preprocess
from features import extract_features
from credibility_model import build_training_labels
from model_zoo import MODEL_SPECS, build_pipeline
from results_io import save_result, load_result
from splits import SEED

SOURCE = "kaggle_full"
SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="10k rows, 3 folds")
    parser.add_argument("--only", help="comma-separated model_ids")
    args = parser.parse_args()

    k = 3 if args.quick else 5
    df = extract_features(preprocess(load_raw(SOURCE, max_rows=10_000 if args.quick else None)))
    df = build_training_labels(df)
    y = df["label"]
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)

    model_ids = args.only.split(",") if args.only else list(MODEL_SPECS)
    previous = load_result("cv_results") if args.only else None
    if previous and (previous["meta"]["n_rows"] != len(df) or previous["data"]["k"] != k):
        previous = None
    results = dict(previous["data"]["models"]) if previous else {}

    for model_id in model_ids:
        start = time.perf_counter()
        scores = cross_validate(build_pipeline(model_id), df, y, cv=cv, scoring=SCORING, n_jobs=-1)
        results[model_id] = {
            "model": MODEL_SPECS[model_id].model,
            "feature_set": MODEL_SPECS[model_id].feature_set,
            **{m: [round(float(v), 4) for v in scores[f"test_{m}"]] for m in SCORING},
            "fit_seconds": [round(float(v), 2) for v in scores["fit_time"]],
        }
        f1 = scores["test_f1"]
        print(f"{model_id:22s} F1 {f1.mean():.4f} ± {f1.std():.4f}  ({time.perf_counter() - start:.0f}s)",
              flush=True)

    path = save_result("cv_results", {"k": k, "models": results}, dataset=SOURCE, n_rows=len(df),
                       script="experiments/run_cross_validation.py")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
