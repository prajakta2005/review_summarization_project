"""
summary_eval.py
---------------
Measures whether credibility-aware summaries are better than standard
ones. All three summaries come from the same TF-IDF ranker, so the only
difference is how much each review's sentences are trusted:

- standard:          every review weighted equally
- credibility_aware: weighted by the model's credibility score (ours)
- oracle:            only ground-truth genuine reviews (the reference)

Metrics (per product/category):
- contamination:    share of summary sentences taken from FAKE reviews (lower is better)
- mean_source_cred: average credibility score of the reviews quoted (higher is better)
- rouge1/2/L:       overlap with the genuine-only oracle summary (higher is better)
- redundancy:       mean pairwise TF-IDF cosine between summary sentences (lower is better)
- coverage:         share of the product's most-discussed attributes the summary mentions
"""

import itertools

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from comparison import find_attribute_mentions
from summarizer import standard_summary, credibility_aware_summary, oracle_summary

VARIANTS = ["standard", "credibility_aware", "oracle"]
_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def contamination_rate(items: list) -> float:
    """items: summary sentences as dicts with an `is_fake` flag."""
    flagged = [it["is_fake"] for it in items if it.get("is_fake") is not None]
    return float(np.mean(flagged)) if flagged else None


def mean_source_credibility(items: list) -> float:
    return float(np.mean([it["score"] for it in items])) if items else None


def rouge_vs_reference(summary: str, reference: str) -> dict:
    if not summary or not reference:
        return {"rouge1": None, "rouge2": None, "rougeL": None}
    scores = _SCORER.score(reference, summary)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}


def redundancy(sentences: list) -> float:
    if len(sentences) < 2:
        return 0.0
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    sims = cosine_similarity(tfidf)
    pairs = [sims[i, j] for i, j in itertools.combinations(range(len(sentences)), 2)]
    return float(np.mean(pairs))


def top_attributes(subset: pd.DataFrame, k: int = 3) -> list:
    counts = pd.Series([a for text in subset["clean_text"] for a in find_attribute_mentions(text)]).value_counts()
    return counts.head(k).index.tolist()


def attribute_coverage(summary: str, attributes: list) -> float:
    if not attributes:
        return None
    mentioned = set(find_attribute_mentions(summary))
    return len(mentioned & set(attributes)) / len(attributes)


def summary_metrics(items: list, reference: str, attributes: list) -> dict:
    text = " ".join(it["sentence"] for it in items)
    return {
        "contamination": contamination_rate(items),
        "mean_source_cred": mean_source_credibility(items),
        **rouge_vs_reference(text, reference),
        "redundancy": round(redundancy([it["sentence"] for it in items]), 4),
        "coverage": attribute_coverage(text, attributes),
    }


def evaluate_summaries(df: pd.DataFrame, product_id: str, top_n: int = 5,
                       weight_floor: float = 0.05, hard_threshold: float = None) -> dict:
    """The three summaries for one product, with their metrics.
    df needs clean_text, credibility_score and ground_truth_label."""
    subset = df[df["product_id"] == product_id]
    summaries = {
        "standard": standard_summary(df, product_id, top_n)["items"],
        "credibility_aware": credibility_aware_summary(df, product_id, top_n, weight_floor=weight_floor,
                                                       hard_threshold=hard_threshold)["overall_items"],
        "oracle": oracle_summary(df, product_id, top_n)["items"],
    }
    reference = " ".join(it["sentence"] for it in summaries["oracle"])
    attributes = top_attributes(subset)
    metrics = {v: summary_metrics(items, reference, attributes) for v, items in summaries.items()}
    return {"summaries": summaries, "metrics": metrics, "attributes": attributes}
