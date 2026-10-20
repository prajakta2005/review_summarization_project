import numpy as np
import pandas as pd

from performance import mcnemar, threshold_sweep, per_category_metrics, per_class_report


def test_threshold_sweep_recall_is_monotonic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    score = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    sweep = threshold_sweep(y, score)
    assert (np.diff(sweep["recall"]) <= 1e-12).all()   # higher threshold -> fewer predicted genuine
    assert sweep["threshold"].between(0, 1).all()
    assert (sweep[["tn", "fp", "fn", "tp"]].sum(axis=1) == 500).all()


def test_threshold_sweep_handles_margins():
    sweep = threshold_sweep([0, 1, 1, 0], [-2.0, 1.5, 0.3, -0.1])
    assert sweep["threshold"].min() == -2.0


def test_mcnemar_identical_predictions():
    y = [0, 1, 1, 0, 1]
    assert mcnemar(y, y, y)["p_value"] == 1.0


def test_mcnemar_detects_clear_difference():
    y = np.array([1] * 200)
    a = np.ones(200, dtype=int)          # always right
    b = np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)]  # wrong half the time
    result = mcnemar(y, a, b)
    assert result["b"] == 100 and result["c"] == 0
    assert result["p_value"] < 1e-10


def test_per_category_and_per_class():
    preds = pd.DataFrame({"review_id": [1, 2, 3, 4], "y_true": [1, 0, 1, 0], "y_pred": [1, 0, 0, 0]})
    cats = pd.Series({1: "A", 2: "A", 3: "B", 4: "B"})
    out = per_category_metrics(preds, cats).set_index("category")
    assert out.loc["A", "accuracy"] == 1.0 and out.loc["B", "accuracy"] == 0.5
    report = per_class_report(preds["y_true"], preds["y_pred"])
    assert list(report.index) == ["fake", "genuine"]
