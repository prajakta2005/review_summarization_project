"""
Trains every model in model_zoo on the shared split and compares them.

    python experiments/run_model_comparison.py                  # all models
    python experiments/run_model_comparison.py --only lr_tfidf,svm_tfidf
    python experiments/run_model_comparison.py --max-rows 5000  # quick dev run

Writes:
  results/model_comparison.json          leaderboard (metrics, curves, timings)
  results/predictions/<model_id>.csv     review_id, y_true, y_pred, score
  models/tfidf_lr.joblib                 text model for LIME explanations (not in git)

DistilBERT is trained on Colab (notebooks/bert_finetune_colab.ipynb). When
results/predictions/distilbert.csv (+ distilbert_meta.json) exist they are
merged into the leaderboard with the same compute_metrics().
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import pandas as pd

from pipeline import load_raw
from preprocessing import preprocess
from features import extract_features
from credibility_model import labeled_split
from evaluation import compute_metrics
from model_zoo import MODEL_SPECS, BERT_SPEC, fit_and_predict, build_text_explainer_model
from results_io import save_result, load_result

SOURCE = "kaggle_full"
PREDICTIONS_DIR = "results/predictions"
TEXT_MODEL_PATH = "models/tfidf_lr.joblib"


def save_predictions(model_id: str, review_ids, y_true, y_pred, score):
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    pd.DataFrame({"review_id": review_ids, "y_true": y_true, "y_pred": y_pred,
                  "score": score}).to_csv(os.path.join(PREDICTIONS_DIR, f"{model_id}.csv"), index=False)


def evaluate_spec(model_id, df, y, train_idx, test_idx) -> dict:
    spec = MODEL_SPECS[model_id]
    out = fit_and_predict(model_id, df, y, train_idx, test_idx)
    save_predictions(model_id, df["review_id"].iloc[test_idx].to_numpy(),
                     out["y_true"], out["y_pred"], out["score"])
    metrics = compute_metrics(out["y_true"], out["y_pred"], out["score"])
    entry = {
        "model_id": model_id, "model": spec.model, "feature_set": spec.feature_set,
        "interpretable": spec.interpretable,
        "score_type": "probability" if hasattr(out["model"], "predict_proba") else "margin",
        "train_seconds": out["train_seconds"], "predict_seconds": out["predict_seconds"],
        "n_train": len(train_idx), "n_test": len(test_idx),
        **metrics,
    }
    return entry, out["model"]


def bert_entry(test_review_ids) -> dict:
    """Leaderboard entry from the Colab predictions, or None if absent."""
    pred_path = os.path.join(PREDICTIONS_DIR, f"{BERT_SPEC.model_id}.csv")
    if not os.path.exists(pred_path):
        print("No DistilBERT predictions found -- run notebooks/bert_finetune_colab.ipynb on Colab "
              f"and save its output to {pred_path}")
        return None
    preds = pd.read_csv(pred_path)
    if set(preds["review_id"]) != set(test_review_ids):
        print(f"WARNING: {pred_path} covers a different test set than the shared split "
              f"({len(preds)} vs {len(test_review_ids)} rows) -- skipping it. "
              "Re-run the notebook with the current results/split_indices.json.")
        return None
    meta_path = os.path.join(PREDICTIONS_DIR, f"{BERT_SPEC.model_id}_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    metrics = compute_metrics(preds["y_true"], preds["y_pred"], preds["score"])
    return {
        "model_id": BERT_SPEC.model_id, "model": BERT_SPEC.model, "feature_set": "bert",
        "interpretable": False, "score_type": "probability",
        "train_seconds": meta.get("train_seconds"), "predict_seconds": meta.get("predict_seconds"),
        "n_train": meta.get("n_train"), "n_test": len(preds),
        "trained_on": meta.get("device", "colab"), "train_subset": meta.get("train_subset"),
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated model_ids (others keep their previous results)")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    df = extract_features(preprocess(load_raw(SOURCE, max_rows=args.max_rows)))
    df, train_idx, test_idx = labeled_split(df)
    y = df["label"]

    model_ids = args.only.split(",") if args.only else list(MODEL_SPECS)
    previous = load_result("model_comparison") if args.only else None
    if previous and previous["meta"]["n_rows"] != len(df):
        print("Previous results used a different number of rows -- not merging them.")
        previous = None
    entries = {m["model_id"]: m for m in previous["data"]["models"]} if previous else {}

    for model_id in [m for m in model_ids if m in MODEL_SPECS]:
        entry, model = evaluate_spec(model_id, df, y, train_idx, test_idx)
        entries[model_id] = entry
        print(f"{model_id:22s} acc={entry['accuracy']:.4f} f1={entry['f1']:.4f} "
              f"auc={entry['roc_auc']} train={entry['train_seconds']}s", flush=True)

    if not args.only or BERT_SPEC.model_id in model_ids:
        bert = bert_entry(df["review_id"].iloc[test_idx])
        if bert:
            entries[bert["model_id"]] = bert
            print(f"{'distilbert':22s} acc={bert['accuracy']:.4f} f1={bert['f1']:.4f} auc={bert['roc_auc']} (from Colab)")
        else:
            entries.pop(BERT_SPEC.model_id, None)

    # text model for LIME word-level explanations (feature 05)
    text_model = build_text_explainer_model()
    text_model.fit(df["clean_text"].iloc[train_idx], y.iloc[train_idx])
    os.makedirs(os.path.dirname(TEXT_MODEL_PATH), exist_ok=True)
    joblib.dump(text_model, TEXT_MODEL_PATH)
    print(f"Saved {TEXT_MODEL_PATH}")

    models = sorted(entries.values(), key=lambda m: m["f1"], reverse=True)
    data = {
        "models": models,
        "best_by_f1": models[0]["model_id"],
        "deployed_model": "rf_handcrafted",
        "n_test": len(test_idx),
    }
    path = save_result("model_comparison", data, dataset=SOURCE, n_rows=len(df),
                       script="experiments/run_model_comparison.py")
    print(f"Wrote {path} (best by F1: {data['best_by_f1']})")


if __name__ == "__main__":
    main()
