import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
# Modules use repo-relative paths (data/, models/, results/)
os.chdir(ROOT)


@pytest.fixture(scope="session")
def sample_raw():
    from pipeline import load_raw
    return load_raw("sample")


@pytest.fixture(scope="session")
def sample_result(sample_raw):
    from pipeline import run_pipeline
    return run_pipeline(sample_raw)


@pytest.fixture(scope="session")
def kaggle_small():
    """1,000-row stratified slice of the real dataset: big enough for
    two classes everywhere, small enough to keep tests fast."""
    from pipeline import load_raw
    from preprocessing import preprocess
    from features import extract_features
    return extract_features(preprocess(load_raw("kaggle_full", max_rows=1000)))
