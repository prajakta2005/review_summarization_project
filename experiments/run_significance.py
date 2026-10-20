"""
McNemar's test between every pair of models, on their saved predictions
for the shared test set (results/predictions/*.csv). Tells us whether
"model A beats model B" is a real difference or noise.

    python experiments/run_significance.py

Writes results/significance.json.
"""

import itertools
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from performance import available_prediction_models, load_predictions, mcnemar
from results_io import save_result

ALPHA = 0.05


def main():
    model_ids = available_prediction_models()
    preds = {m: load_predictions(m).sort_values("review_id").reset_index(drop=True) for m in model_ids}
    n_test = len(next(iter(preds.values())))

    pairs = []
    for a, b in itertools.combinations(model_ids, 2):
        pa, pb = preds[a], preds[b]
        if not pa["review_id"].equals(pb["review_id"]):
            print(f"skipping {a} vs {b}: different test rows")
            continue
        test = mcnemar(pa["y_true"], pa["y_pred"], pb["y_pred"])
        acc_a = float((pa["y_pred"] == pa["y_true"]).mean())
        acc_b = float((pb["y_pred"] == pb["y_true"]).mean())
        pairs.append({"model_a": a, "model_b": b, "accuracy_a": round(acc_a, 4), "accuracy_b": round(acc_b, 4),
                      **test, "significant": test["p_value"] < ALPHA})

    path = save_result("significance", {"alpha": ALPHA, "n_test": n_test, "pairs": pairs},
                       dataset="kaggle_full", n_rows=n_test, script="experiments/run_significance.py")
    print(f"Wrote {path}: {len(pairs)} pairs, {sum(p['significant'] for p in pairs)} significant at {ALPHA}")


if __name__ == "__main__":
    main()
