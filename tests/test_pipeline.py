import numpy as np
import pandas as pd
import pytest

from pipeline import adapt_schema, load_raw
from splits import train_test_indices


def test_pipeline_runs_on_sample(sample_result):
    explained = sample_result.explained
    for col in ["credibility_score", "credibility_label", "explanation_text",
                "positive_factors", "negative_factors"]:
        assert col in explained.columns
    assert explained["credibility_score"].between(0, 100).all()


def test_pipeline_split_covers_all_rows(sample_result):
    n = len(sample_result.df)
    both = np.concatenate([sample_result.train_idx, sample_result.test_idx])
    assert sorted(both) == list(range(n))


def test_split_is_disjoint_and_deterministic():
    y = pd.Series([0, 1] * 50)
    tr1, te1 = train_test_indices(y)
    tr2, te2 = train_test_indices(y)
    assert set(tr1).isdisjoint(te1)
    assert len(tr1) + len(te1) == len(y)
    assert (tr1 == tr2).all() and (te1 == te2).all()
    # stratified: both classes equally represented in test
    assert y.iloc[te1].value_counts().nunique() == 1


def test_split_handles_single_class():
    tr, te = train_test_indices(pd.Series([1] * 10))
    assert len(tr) + len(te) == 10


def test_kaggle_downsample_is_stratified():
    df = load_raw("kaggle_full", max_rows=200)
    assert len(df) == 200
    assert df["ground_truth_label"].value_counts().tolist() == [100, 100]


def test_adapt_schema_rejects_unknown_columns():
    with pytest.raises(ValueError):
        adapt_schema(pd.DataFrame({"foo": [1]}))
