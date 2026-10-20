"""
Evaluates the three explanation methods for the deployed Random Forest
on the full dataset.

    python experiments/run_explainability_eval.py            # 500 test reviews (~2 min)
    python experiments/run_explainability_eval.py --n 100    # quicker

Writes results/explainability_eval.json: global mean |SHAP|, genuine-vs-
fake SHAP contrast, method agreement, deletion faithfulness, examples.
The ablation script reads `global_shap` for its ranking comparison.
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from pipeline import load_raw
from preprocessing import preprocess
from features import extract_features, FEATURE_COLUMNS
from credibility_model import train_model, labeled_split
from shap_explainer import shap_values, base_value
from lime_explainer import lime_tabular_explain, lime_text_explain, load_text_model
from explanation_eval import (ranking_from_scores, zscore_scores, method_agreement,
                              deletion_faithfulness, class_contrast)
from results_io import save_result
from splits import SEED

SOURCE = "kaggle_full"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="test reviews to explain with all methods")
    args = parser.parse_args()

    df = extract_features(preprocess(load_raw(SOURCE)))
    clf, _, _ = train_model(df)
    df, train_idx, test_idx = labeled_split(df)
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    # global SHAP + class contrast on the whole test split
    test_shap = shap_values(clf, test)
    global_shap = {f: round(float(v), 5) for f, v in zip(FEATURE_COLUMNS, np.abs(test_shap).mean(axis=0))}
    contrast = class_contrast(test_shap, test["label"])
    print("mean |SHAP|:", dict(sorted(global_shap.items(), key=lambda kv: -kv[1])))

    # per-review rankings from every method on a sample of test reviews
    sample = test.sample(min(args.n, len(test)), random_state=SEED)
    sample_shap = shap_values(clf, sample)
    means, stds = train[FEATURE_COLUMNS].mean(), train[FEATURE_COLUMNS].std()
    lime_background = train.sample(min(5000, len(train)), random_state=SEED)

    rankings = {"zscore": [], "shap": [], "lime": []}
    for i, (_, row) in enumerate(sample.iterrows()):
        rankings["zscore"].append(ranking_from_scores(zscore_scores(row, means, stds)))
        rankings["shap"].append(ranking_from_scores(dict(zip(FEATURE_COLUMNS, sample_shap[i]))))
        lime = lime_tabular_explain(clf, lime_background, row)
        rankings["lime"].append(ranking_from_scores({d["feature"]: d["weight"] for d in lime}))
        if (i + 1) % 100 == 0:
            print(f"  explained {i + 1}/{len(sample)}", flush=True)

    agreement = method_agreement(rankings)
    faithfulness = deletion_faithfulness(clf, sample, rankings, means)
    print("top-1 agreement:", agreement["top1_match"])
    print("deletion test (k=1,2,3):", faithfulness["prob_change"])

    # a few worked examples for the dashboard / report
    text_model = load_text_model(train["clean_text"], train["label"])
    examples = []
    for label_value, name in [(0, "fake"), (1, "genuine")]:
        picks = sample[sample["label"] == label_value].head(3)
        for _, row in picks.iterrows():
            i = sample.index.get_loc(row.name)
            examples.append({
                "review_id": int(row["review_id"]), "label": name,
                "score": round(float(clf.predict_proba(row[FEATURE_COLUMNS].to_frame().T)[0, 1]), 4),
                "text": str(row["clean_text"])[:500],
                "shap": {f: round(float(v), 4) for f, v in zip(FEATURE_COLUMNS, sample_shap[i])},
                "lime_words": [[w, round(v, 4)] for w, v in lime_text_explain(text_model, row["clean_text"])],
            })

    data = {
        "n_explained": len(sample),
        "base_value": round(base_value(clf), 4),
        "global_shap": global_shap,
        "class_contrast": contrast,
        "agreement": agreement,
        "faithfulness": faithfulness,
        "examples": examples,
    }
    path = save_result("explainability_eval", data, dataset=SOURCE, n_rows=len(df),
                       script="experiments/run_explainability_eval.py")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
