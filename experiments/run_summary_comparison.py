"""
Standard vs credibility-aware summarization on the full dataset.

    python experiments/run_summary_comparison.py           # ~4 min
    python experiments/run_summary_comparison.py --quick   # 6k rows, top_n=5 only

Leakage control: every review's credibility score comes from a model that
never saw that review (5-fold out-of-fold predictions). Scoring reviews
with a model trained on them would make the improvement look better
than it is.

Writes results/summary_comparison.json.
"""

import argparse
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from pipeline import load_raw
from preprocessing import preprocess
from features import extract_features, FEATURE_COLUMNS
from credibility_model import build_training_labels, LOW_THRESHOLD
from model_zoo import build_pipeline
from summary_eval import evaluate_summaries
from results_io import save_result
from splits import SEED

SOURCE = "kaggle_full"
TOP_N = [3, 5, 8]
WEIGHT_FLOORS = [0.0, 0.05, 0.2, 0.5]
HARD_THRESHOLDS = [40, 50, 60]
METRICS = ["contamination", "mean_source_cred", "rouge1", "rouge2", "rougeL", "redundancy", "coverage"]


def out_of_fold_scores(df: pd.DataFrame) -> np.ndarray:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(build_pipeline("rf_handcrafted"), df[FEATURE_COLUMNS], df["label"],
                              cv=cv, method="predict_proba")
    return proba[:, 1] * 100


def rows_for(result: dict, category: str, top_n: int, setting: dict, variants=("standard", "credibility_aware")):
    return [{"category": category, "top_n": top_n, "variant": v, **setting,
             **{m: result["metrics"][v][m] for m in METRICS}} for v in variants]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    top_ns = [5] if args.quick else TOP_N

    df = build_training_labels(extract_features(preprocess(load_raw(SOURCE, max_rows=6000 if args.quick else None))))
    start = time.perf_counter()
    df["credibility_score"] = out_of_fold_scores(df).round(1)
    print(f"out-of-fold credibility scores: {time.perf_counter() - start:.0f}s", flush=True)

    categories = sorted(df["product_id"].unique())
    main_rows, floor_rows, threshold_rows, examples = [], [], [], []
    for category in categories:
        for top_n in top_ns:
            result = evaluate_summaries(df, category, top_n)
            main_rows += rows_for(result, category, top_n, {"weight_floor": 0.05})
            if top_n == 5:
                examples.append({"category": category, "top_n": top_n, "attributes": result["attributes"],
                                 **{v: result["summaries"][v] for v in result["summaries"]}})
        for floor in WEIGHT_FLOORS:
            result = evaluate_summaries(df, category, 5, weight_floor=floor)
            floor_rows += rows_for(result, category, 5, {"weight_floor": floor}, variants=("credibility_aware",))
        for threshold in HARD_THRESHOLDS:
            result = evaluate_summaries(df, category, 5, hard_threshold=threshold)
            threshold_rows += rows_for(result, category, 5, {"hard_threshold": threshold},
                                       variants=("credibility_aware",))
        print(f"  {category} done", flush=True)

    main_df = pd.DataFrame(main_rows)
    averages = {
        str(top_n): main_df[main_df["top_n"] == top_n].groupby("variant")[METRICS].mean().round(4).to_dict(orient="index")
        for top_n in top_ns
    }
    sensitivity = {
        "weight_floor": pd.DataFrame(floor_rows).groupby("weight_floor")[METRICS].mean().round(4).reset_index().to_dict(orient="records"),
        "hard_threshold": pd.DataFrame(threshold_rows).groupby("hard_threshold")[METRICS].mean().round(4).reset_index().to_dict(orient="records"),
    }

    headline = averages["5"]
    print("top_n=5 averages over categories:")
    for variant, m in headline.items():
        print(f"  {variant:18s} contamination {m['contamination']:.3f}  ROUGE-L {m['rougeL']:.3f}  "
              f"mean source credibility {m['mean_source_cred']:.1f}")

    data = {
        "score_source": "out_of_fold_5cv",
        "settings": {"top_n": top_ns, "weight_floor": WEIGHT_FLOORS, "hard_threshold": HARD_THRESHOLDS,
                     "default_weight_floor": 0.05, "low_threshold": LOW_THRESHOLD},
        "n_categories": len(categories),
        "per_category": main_rows,
        "averages": averages,
        "sensitivity": sensitivity,
        "examples": examples,
    }
    path = save_result("summary_comparison", data, dataset=SOURCE, n_rows=len(df),
                       script="experiments/run_summary_comparison.py")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
