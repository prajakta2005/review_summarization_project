import pytest

from ablation import cv_score, leave_one_out, group_ablation, forward_selection
from features import FEATURE_COLUMNS, FEATURE_GROUPS


@pytest.fixture(scope="module")
def labeled(kaggle_small):
    from credibility_model import build_training_labels
    df = build_training_labels(kaggle_small)
    return df, df["label"]


def test_feature_groups_cover_every_feature_once():
    grouped = [f for members in FEATURE_GROUPS.values() for f in members]
    assert sorted(grouped) == sorted(FEATURE_COLUMNS)


def test_leave_one_out_drops_each_feature(labeled):
    df, y = labeled
    full = cv_score(df, y, "lr_handcrafted", FEATURE_COLUMNS, k=3)
    loo = leave_one_out(df, y, "lr_handcrafted", full, k=3)
    assert sorted(r["removed"] for r in loo) == sorted(FEATURE_COLUMNS)
    assert all(len(r["f1"]["folds"]) == 3 for r in loo)


def test_group_ablation(labeled):
    df, y = labeled
    full = cv_score(df, y, "lr_handcrafted", FEATURE_COLUMNS, k=3)
    groups = group_ablation(df, y, "lr_handcrafted", full, k=3)
    assert {g["removed_group"] for g in groups} == set(FEATURE_GROUPS)


def test_forward_selection_adds_every_feature(labeled):
    df, y = labeled
    steps = forward_selection(df, y, "lr_handcrafted", k=2)
    assert [s["step"] for s in steps] == list(range(1, len(FEATURE_COLUMNS) + 1))
    assert sorted(steps[-1]["features"]) == sorted(FEATURE_COLUMNS)
