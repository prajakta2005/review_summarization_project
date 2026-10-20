"""
Ablation study on the 6 handcrafted credibility features, for the
deployed Random Forest and for Logistic Regression.

    python experiments/run_ablation.py           # full dataset, 5-fold CV (a few minutes)
    python experiments/run_ablation.py --quick   # 10k rows, 3 folds (< 1 min)

Experiments: A leave-one-out, B single-feature, C greedy forward addition,
D feature-group removal, E representation (from model_comparison.json),
plus a comparison of importance rankings (ablation vs impurity vs
permutation vs SHAP if results/explainability_eval.json exists).

Writes results/ablation.json.
"""

import argparse
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import load_raw
from preprocessing import preprocess
from features import extract_features, FEATURE_COLUMNS, FEATURE_GROUPS
from credibility_model import labeled_split
from ablation import (ABLATION_MODELS, cv_score, leave_one_out, single_feature,
                      forward_selection, group_ablation, importance_rankings)
from results_io import save_result, load_result

SOURCE = "kaggle_full"


def representation_ablation() -> dict:
    """E: does adding the handcrafted features to TF-IDF help? (from 02)"""
    comparison = load_result("model_comparison")
    if not comparison:
        return None
    by_id = {m["model_id"]: m for m in comparison["data"]["models"]}
    out = {}
    for family in ["lr", "svm"]:
        tfidf, hybrid = by_id.get(f"{family}_tfidf"), by_id.get(f"{family}_hybrid")
        if tfidf and hybrid:
            out[family] = {"tfidf_f1": tfidf["f1"], "hybrid_f1": hybrid["f1"],
                           "delta_f1": round(hybrid["f1"] - tfidf["f1"], 4)}
    significance = load_result("significance")
    if significance:
        for p in significance["data"]["pairs"]:
            for family in out:
                if {p["model_a"], p["model_b"]} == {f"{family}_tfidf", f"{family}_hybrid"}:
                    out[family]["p_value"] = round(p["p_value"], 4)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="10k rows, 3 folds")
    args = parser.parse_args()
    k = 3 if args.quick else 5

    df = extract_features(preprocess(load_raw(SOURCE, max_rows=10_000 if args.quick else None)))
    df, train_idx, test_idx = labeled_split(df)
    y = df["label"]

    models = {}
    for name, model_id in ABLATION_MODELS.items():
        start = time.perf_counter()
        full = cv_score(df, y, model_id, FEATURE_COLUMNS, k)
        loo = leave_one_out(df, y, model_id, full, k)
        models[name] = {
            "model_id": model_id,
            "full": full,
            "leave_one_out": loo,
            "single_feature": single_feature(df, y, model_id, k),
            "forward": forward_selection(df, y, model_id, k),
            "groups": group_ablation(df, y, model_id, full, k),
        }
        print(f"{name}: full F1 {full['f1']['mean']:.4f}; biggest drop without "
              f"{loo[0]['removed']} ({loo[0]['delta_f1']:+.4f})  [{time.perf_counter() - start:.0f}s]", flush=True)

    explain = load_result("explainability_eval")
    shap_importance = explain["data"].get("global_shap") if explain else None
    rankings = importance_rankings(df, y, train_idx, test_idx, models["rf"]["leave_one_out"], shap_importance)
    print("Spearman vs ablation ranking:", rankings["spearman_vs_ablation"])

    data = {
        "k": k, "metric_primary": "f1",
        "feature_groups": FEATURE_GROUPS,
        "models": models,
        "rankings": rankings,
        "representation": representation_ablation(),
    }
    path = save_result("ablation", data, dataset=SOURCE, n_rows=len(df), script="experiments/run_ablation.py")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
