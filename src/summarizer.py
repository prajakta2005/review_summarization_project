"""
summarizer.py
-------------
Credibility-AWARE summarization -- the module that differentiates this
project from a generic review summarizer.

Approach: EXTRACTIVE summarization (TF-IDF sentence scoring + picking
top-ranked real sentences), not abstractive/generative. Two reasons:
1. No internet access in this sandbox to pull a pretrained abstractive
   model (BART/T5/etc from HuggingFace) or call an external LLM API
   with cost/keys.
2. Extractive summaries are auditable -- every sentence in the output
   is a REAL sentence from a REAL review, which matters for a
   credibility-focused system (you don't want your summarizer
   hallucinating claims about a product).

Upgrade path for your report: swap `rank_sentences()` for a call to
an abstractive model or LLM API once you have one available; the
credibility-weighting logic around it does not need to change.

Credibility-aware behavior (the actual novelty):
- Sentences from HIGH credibility reviews are weighted up.
- Sentences from LOW credibility reviews are weighted down (not
  deleted -- per the spec, low-credibility content should be
  surfaced separately, not silently discarded).
- Produces three views: overall summary, high-credibility consensus,
  and a flag list of low-credibility claims for transparency.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from credibility_model import HIGH_THRESHOLD, LOW_THRESHOLD


def split_sentences(text: str):
    # Simple regex sentence splitter (no internet access for nltk's
    # punkt tokenizer download in this sandbox). Good enough for
    # review-length text; swap for nltk.sent_tokenize with internet access.
    sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
    return [s.strip() for s in sentences if len(s.strip()) > 3]


def rank_sentences(sentences: list, weights: list, top_n: int, return_indices: bool = False) -> list:
    """TF-IDF cosine-similarity-to-centroid ranking, scaled by
    per-sentence credibility weight. Returns top_n sentences in
    their original relative order (reads more naturally).
    return_indices=True returns their positions instead, so each summary
    sentence can be traced back to the review it came from.
    """
    if not sentences:
        return []
    if len(sentences) <= top_n:
        top_idx = list(range(len(sentences)))
    else:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(sentences)
        centroid = np.asarray(tfidf.mean(axis=0))
        scores = np.asarray(tfidf @ centroid.T).flatten()
        weighted_scores = scores * np.array(weights)
        top_idx = sorted(int(i) for i in np.argsort(-weighted_scores)[:top_n])  # keep original order
    return top_idx if return_indices else [sentences[i] for i in top_idx]


def _collect_sentences(subset: pd.DataFrame) -> pd.DataFrame:
    """One row per sentence: sentence, review_id, credibility score of its
    review, and is_fake from the ground-truth label (None if unlabeled)."""
    has_labels = "ground_truth_label" in subset.columns and subset["ground_truth_label"].notna().all()
    rows = []
    for _, row in subset.iterrows():
        for sent in split_sentences(row["clean_text"]):
            rows.append({
                "sentence": sent,
                "review_id": row["review_id"],
                "score": row["credibility_score"],
                "is_fake": bool(row["ground_truth_label"] == 0) if has_labels else None,
            })
    return pd.DataFrame(rows, columns=["sentence", "review_id", "score", "is_fake"])


def _summarize(sentences: pd.DataFrame, weights, top_n: int) -> list:
    """Ranks the collected sentences with the given weights and returns the
    chosen ones as dicts (sentence, review_id, score, is_fake)."""
    if sentences.empty:
        return []
    idx = rank_sentences(sentences["sentence"].tolist(), list(weights), top_n, return_indices=True)
    return sentences.iloc[idx].to_dict(orient="records")


def _subset(df: pd.DataFrame, product_id: str = None) -> pd.DataFrame:
    return df if product_id is None else df[df["product_id"] == product_id]


def standard_summary(df: pd.DataFrame, product_id: str = None, top_n: int = 4) -> dict:
    """BASELINE: the same ranker with every review weighted equally, i.e.
    a normal review summarizer that trusts fake and genuine reviews alike."""
    sentences = _collect_sentences(_subset(df, product_id))
    items = _summarize(sentences, [1.0] * len(sentences), top_n)
    return {"summary": " ".join(it["sentence"] for it in items), "items": items}


def oracle_summary(df: pd.DataFrame, product_id: str = None, top_n: int = 4) -> dict:
    """REFERENCE: the same ranker over ground-truth GENUINE reviews only --
    what a perfect credibility filter would give. Needs ground_truth_label."""
    subset = _subset(df, product_id)
    if "ground_truth_label" not in subset.columns:
        return {"summary": "", "items": []}
    sentences = _collect_sentences(subset[subset["ground_truth_label"] == 1])
    items = _summarize(sentences, [1.0] * len(sentences), top_n)
    return {"summary": " ".join(it["sentence"] for it in items), "items": items}


def credibility_aware_summary(df: pd.DataFrame, product_id: str = None, top_n: int = 4,
                              weight_floor: float = 0.05, hard_threshold: float = None) -> dict:
    """df must already have `clean_text` and `credibility_score` columns.
    Returns overall summary, high-credibility consensus, and flagged
    low-credibility claims -- for one product if product_id given, else all.

    weight_floor: minimum weight for a sentence (so low-credibility text is
      down-weighted, never fully zeroed out).
    hard_threshold: optional score (0-100); reviews below it are dropped
      from the overall summary entirely (a hard filter, for comparison).
    "overall_items" / "sources" trace each summary sentence to its review.
    """
    subset = _subset(df, product_id)
    if subset.empty:
        return {"overall_summary": "", "overall_items": [], "sources": [],
                "high_credibility_consensus": "", "low_credibility_flags": []}

    sentences = _collect_sentences(subset)
    all_sentences = sentences["sentence"].tolist()
    sentence_source_credibility = sentences["score"].tolist()

    # Overall summary: all sentences, credibility-weighted
    ranked = sentences if hard_threshold is None else sentences[sentences["score"] >= hard_threshold]
    overall_items = _summarize(ranked, (ranked["score"] / 100).clip(lower=weight_floor), top_n)
    overall = [it["sentence"] for it in overall_items]

    # High-credibility-only consensus (>= HIGH_THRESHOLD score)
    high_cred_sentences = [s for s, c in zip(all_sentences, sentence_source_credibility) if c >= HIGH_THRESHOLD]
    high_cred_weights = [1.0] * len(high_cred_sentences)
    consensus = rank_sentences(high_cred_sentences, high_cred_weights, top_n)

    # Low-credibility flags: surfaced, not deleted, per the spec
    low_cred_rows = subset[subset["credibility_score"] < LOW_THRESHOLD]
    flags = [
        {"review_id": r["review_id"], "credibility_score": r["credibility_score"],
         "excerpt": r["clean_text"][:120]}
        for _, r in low_cred_rows.iterrows()
    ]

    return {
        "overall_summary": " ".join(overall),
        "overall_items": overall_items,
        "sources": [it["review_id"] for it in overall_items],
        "high_credibility_consensus": " ".join(consensus) if consensus else "Not enough high-credibility reviews yet.",
        "low_credibility_flags": flags,
    }


if __name__ == "__main__":
    from preprocessing import load_reviews, preprocess
    from features import extract_features
    from credibility_model import train_model, score_reviews

    df = load_reviews("data/sample_reviews.csv")
    df = preprocess(df)
    df = extract_features(df)
    clf, _, _ = train_model(df)
    scored = score_reviews(df, clf)

    for pid in scored["product_id"].unique():
        result = credibility_aware_summary(scored, product_id=pid)
        print(f"\n=== {pid} ===")
        print("Overall summary:", result["overall_summary"])
        print("High-credibility consensus:", result["high_credibility_consensus"])
        print("Low-credibility flags:", result["low_credibility_flags"])
