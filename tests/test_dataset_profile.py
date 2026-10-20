import json

from dataset_profile import profile_dataset, cohens_d
from features import FEATURE_COLUMNS
import pandas as pd


def test_profile_keys_and_counts(kaggle_small):
    p = profile_dataset(kaggle_small)
    assert p["labeled"]
    assert sum(p["label_counts"].values()) == len(kaggle_small)
    assert sum(c["total"] for c in p["categories"]) == len(kaggle_small)
    assert set(p["feature_stats_by_label"]) == set(FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        hist = p["feature_hist_by_label"][col]
        assert len(hist["bins"]) == len(hist["genuine"]) + 1
    json.dumps(p)  # must be serializable for results/


def test_profile_on_unlabeled_sample(sample_result):
    p = profile_dataset(sample_result.df)
    assert not p["labeled"]
    assert p["label_counts"] == {"unlabeled": len(sample_result.df)}


def test_cohens_d_sign():
    assert cohens_d(pd.Series([5.0, 6, 7]), pd.Series([1.0, 2, 3])) > 0
    assert cohens_d(pd.Series([1.0]), pd.Series([2.0])) == 0.0
