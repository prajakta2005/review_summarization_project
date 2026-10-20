"""
splits.py
---------
The single train/test split every model and experiment uses.

Comparing models only means something if they are all tested on the
SAME held-out reviews. Every module that needs a split calls
`train_test_indices()` instead of calling sklearn's train_test_split
with its own settings. The BERT Colab notebook can't import this, so
`export_split()` writes the split to disk by review_id.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
SEED = 42
SPLIT_PATH = "results/split_indices.json"


def train_test_indices(y: pd.Series, test_size: float = TEST_SIZE, seed: int = SEED):
    """Stratified split on the label. Returns (train_idx, test_idx) as
    numpy arrays of POSITIONAL indices (use with .iloc). Falls back to an
    unstratified split when y has a single class or a class too small to
    stratify (the 15-row sample dataset).
    """
    positions = np.arange(len(y))
    counts = pd.Series(y).value_counts()
    can_stratify = len(counts) > 1 and counts.min() >= 2
    train_idx, test_idx = train_test_split(
        positions, test_size=test_size, random_state=seed,
        stratify=y if can_stratify else None,
    )
    return np.sort(train_idx), np.sort(test_idx)


def export_split(df: pd.DataFrame, label_col: str = "ground_truth_label", path: str = SPLIT_PATH) -> str:
    """Writes {"train": [review_id, ...], "test": [...]} so code outside
    this repo (the BERT notebook) can reproduce the identical split.
    review_id is used rather than the positional index because
    preprocess() drops duplicates and resets the index.
    """
    train_idx, test_idx = train_test_indices(df[label_col])
    payload = {
        "test_size": TEST_SIZE,
        "seed": SEED,
        "n_rows": len(df),
        "train": df["review_id"].iloc[train_idx].astype(int).tolist(),
        "test": df["review_id"].iloc[test_idx].astype(int).tolist(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
    return path
