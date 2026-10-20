"""
text_features.py
----------------
Feature sets for the model comparison, as sklearn ColumnTransformers
that take the featurized review DataFrame directly:

- handcrafted: the 6 interpretable credibility features (features.py)
- tfidf:       word 1-2-gram TF-IDF of clean_text
- hybrid:      both side by side (tests whether the interpretable
               signals add anything on top of the words)

Because the transformer is part of each model's Pipeline, it is fit on
the training rows only -- no vocabulary leakage from the test set.
"""

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS

FEATURE_SETS = ["handcrafted", "tfidf", "hybrid"]
TEXT_COLUMN = "clean_text"


def build_tfidf(max_features: int = 50_000) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2), max_features=max_features, min_df=2,
        sublinear_tf=True, strip_accents="unicode",
    )


def build_feature_transformer(feature_set: str, scale: bool = True,
                              columns: list = None) -> ColumnTransformer:
    """columns: subset of FEATURE_COLUMNS for the handcrafted part
    (the ablation study passes reduced lists). scale=False for trees."""
    columns = list(columns or FEATURE_COLUMNS)
    handcrafted = ("handcrafted", StandardScaler() if scale else "passthrough", columns)
    tfidf = ("tfidf", build_tfidf(), TEXT_COLUMN)
    if feature_set == "handcrafted":
        parts = [handcrafted]
    elif feature_set == "tfidf":
        parts = [tfidf]
    elif feature_set == "hybrid":
        parts = [tfidf, handcrafted]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return ColumnTransformer(parts, remainder="drop")
