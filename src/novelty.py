"""
novelty.py
----------
Backs the Novelty page: a capability matrix comparing this system with
the single-purpose approaches it combines, evidence cards built from the
experiment results, and the "trace a summary sentence" helper that shows
where one summary sentence came from.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from results_summary import headline_numbers
from summarizer import _collect_sentences

CAPABILITIES = ["Detects fake reviews", "Explains each decision", "Summarizes reviews",
                "Credibility-weighted summary", "Sentence → review → reasons trace",
                "Measures fake content in summaries"]


def capability_matrix() -> pd.DataFrame:
    """✓ / ✗ per approach. The first three rows are families of prior work
    (representative references in docs/NOVELTY.md); verify against the
    closest works you find before claiming anything is new."""
    rows = {
        "Fake-review detection (e.g. Ott et al. 2011; Salminen et al. 2022)": [1, 0, 0, 0, 0, 0],
        "Review summarization (e.g. Hu & Liu 2004)": [0, 0, 1, 0, 0, 0],
        "Explainable ML (LIME 2016, SHAP 2017)": [0, 1, 0, 0, 0, 0],
        "This system": [1, 1, 1, 1, 1, 1],
    }
    return pd.DataFrame({k: ["✓" if v else "✗" for v in vals] for k, vals in rows.items()},
                        index=CAPABILITIES).T


def evidence_cards(results: dict = None) -> list:
    """[{label, value, detail}] for the Novelty page; 'pending' when an
    experiment hasn't been run yet."""
    h = headline_numbers(results)
    pending = {"value": "pending", "detail": "run experiments/run_all.py"}
    cards = []
    if "contamination_ours" in h:
        cards.append({"label": "Fake content in summaries",
                      "value": f"{h['contamination_standard']:.0%} → {h['contamination_ours']:.0%}",
                      "detail": "standard → credibility-aware (5 sentences, avg. over categories)"})
        cards.append({"label": "ROUGE-L vs genuine-only summary",
                      "value": f"{h['rougeL_ours'] - h['rougeL_standard']:+.3f}",
                      "detail": f"{h['rougeL_standard']:.3f} → {h['rougeL_ours']:.3f}"})
    else:
        cards += [{"label": "Fake content in summaries", **pending},
                  {"label": "ROUGE-L vs genuine-only summary", **pending}]
    if "shap_faithfulness_k1" in h:
        cards.append({"label": "Explanation faithfulness (SHAP)",
                      "value": f"{h['shap_faithfulness_k1']:.3f}",
                      "detail": f"prediction change when the top feature is reset; random feature: "
                                f"{h['random_faithfulness_k1']:.3f}"})
    else:
        cards.append({"label": "Explanation faithfulness (SHAP)", **pending})
    if "best_f1" in h:
        cards.append({"label": "Cost of interpretability",
                      "value": f"{h['deployed_f1'] - h['best_f1']:+.3f} F1",
                      "detail": f"explainable RF ({h['deployed_f1']:.3f}) vs best model ({h['best_f1']:.3f})"})
    else:
        cards.append({"label": "Cost of interpretability", **pending})
    return cards


def standard_rank(df: pd.DataFrame, product_id: str, review_id, sentence: str):
    """Rank (1 = best) a sentence would get in the STANDARD summary of its
    product, i.e. the same TF-IDF centroid ranking with uniform weights.
    Returns (rank, n_sentences)."""
    sentences = _collect_sentences(df[df["product_id"] == product_id])
    if sentences.empty:
        return None, 0
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(sentences["sentence"])
    centroid = np.asarray(tfidf.mean(axis=0))
    scores = np.asarray(tfidf @ centroid.T).flatten()
    order = np.argsort(-scores)
    match = np.where((sentences["review_id"].to_numpy() == review_id) &
                     (sentences["sentence"].to_numpy() == sentence))[0]
    if not len(match):
        return None, len(sentences)
    rank = int(np.where(order == match[0])[0][0]) + 1
    return rank, len(sentences)


LIMITATIONS = [
    "In this dataset `product_id` is the Amazon category, so \"product\" summaries are category-level.",
    "\"Fake\" means computer-generated (CG) reviews; human-written deceptive reviews may behave differently.",
    "Summaries are extractive (real sentences, nothing generated), which keeps them traceable but less fluent.",
    "The deployed 6-feature model trades accuracy for explainability; text models score higher (see Model Comparison).",
    "ROUGE against a genuine-only extractive summary is a proxy for quality, not a human judgement.",
]
