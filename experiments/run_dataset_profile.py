"""
Profiles the full Kaggle dataset and exports the shared train/test split.

    python experiments/run_dataset_profile.py

Writes results/dataset_profile.json and results/split_indices.json
(the split file is what the BERT Colab notebook uses).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import load_raw
from preprocessing import preprocess
from features import extract_features
from dataset_profile import profile_dataset
from results_io import save_result
from splits import export_split

SOURCE = "kaggle_full"


def main():
    df = extract_features(preprocess(load_raw(SOURCE)))
    profile = profile_dataset(df)
    path = save_result("dataset_profile", profile, dataset=SOURCE, n_rows=len(df),
                       script="experiments/run_dataset_profile.py")
    print(f"Wrote {path}: {profile['n_after_preprocess']} reviews, {profile['label_counts']}")
    split_path = export_split(df)
    print(f"Wrote {split_path}")


if __name__ == "__main__":
    main()
