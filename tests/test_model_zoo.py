import numpy as np
import pytest

from credibility_model import labeled_split
from evaluation import compute_metrics
from model_zoo import MODEL_SPECS, fit_and_predict, build_text_explainer_model


@pytest.fixture(scope="module")
def split(kaggle_small):
    return labeled_split(kaggle_small)


@pytest.mark.parametrize("model_id", list(MODEL_SPECS))
def test_every_model_fits_and_predicts(split, model_id):
    df, train_idx, test_idx = split
    out = fit_and_predict(model_id, df, df["label"], train_idx, test_idx)
    assert len(out["y_pred"]) == len(out["score"]) == len(test_idx)
    assert set(np.unique(out["y_pred"])) <= {0, 1}
    m = compute_metrics(out["y_true"], out["y_pred"], out["score"])
    assert 0.5 < m["accuracy"] <= 1.0  # better than chance on real data


def test_handcrafted_subset_of_columns(split):
    df, train_idx, test_idx = split
    out = fit_and_predict("lr_handcrafted", df, df["label"], train_idx, test_idx,
                          columns=["length", "specificity"])
    assert out["model"].named_steps["features"].transformers_[0][2] == ["length", "specificity"]


def test_text_explainer_model_takes_raw_strings(split):
    df, train_idx, _ = split
    model = build_text_explainer_model().fit(df["clean_text"].iloc[train_idx], df["label"].iloc[train_idx])
    proba = model.predict_proba(["Great product, works as described.", "Terrible, broke in a day."])
    assert proba.shape == (2, 2)


def test_compute_metrics_confusion_matrix_is_2x2_even_with_one_class():
    m = compute_metrics([1, 1, 1], [1, 1, 1])
    assert m["confusion_matrix"] == [[0, 0], [0, 3]]
    assert m["roc_auc"] is None
