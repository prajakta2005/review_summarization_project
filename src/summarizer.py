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


def split_sentences(text: str):
    # Simple regex sentence splitter (no internet access for nltk's
    # punkt tokenizer download in this sandbox). Good enough for
    # review-length text; swap for nltk.sent_tokenize with internet access.
    sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
    return [s.strip() for s in sentences if len(s.strip()) > 3]


def rank_sentences(sentences: list, weights: list, top_n: int) -> list:
    """TF-IDF cosine-similarity-to-centroid ranking, scaled by
    per-sentence credibility weight. Returns top_n sentences in
    their original relative order (reads more naturally).
    """
    if not sentences:
        return []
    if len(sentences) <= top_n:
        return sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(sentences)
    centroid = np.asarray(tfidf.mean(axis=0))
    scores = np.asarray(tfidf @ centroid.T).flatten()
    weighted_scores = scores * np.array(weights)

    top_idx = sorted(np.argsort(-weighted_scores)[:top_n])  # keep original order
    return [sentences[i] for i in top_idx]


def credibility_aware_summary(df: pd.DataFrame, product_id: str = None, top_n: int = 4) -> dict:
    """df must already have `clean_text` and `credibility_score` columns.
    Returns overall summary, high-credibility consensus, and flagged
    low-credibility claims -- for one product if product_id given, else all.
    """
    subset = df if product_id is None else df[df["product_id"] == product_id]
    if subset.empty:
        return {"overall_summary": "", "high_credibility_consensus": "", "low_credibility_flags": []}

    all_sentences, all_weights, sentence_source_credibility = [], [], []
    for _, row in subset.iterrows():
        for sent in split_sentences(row["clean_text"]):
            all_sentences.append(sent)
            weight = max(row["credibility_score"] / 100, 0.05)  # never fully zero out
            all_weights.append(weight)
            sentence_source_credibility.append(row["credibility_score"])

    # Overall summary: all sentences, credibility-weighted
    overall = rank_sentences(all_sentences, all_weights, top_n)

    # High-credibility-only consensus (>=70 score)
    high_cred_sentences = [s for s, c in zip(all_sentences, sentence_source_credibility) if c >= 70]
    high_cred_weights = [1.0] * len(high_cred_sentences)
    consensus = rank_sentences(high_cred_sentences, high_cred_weights, top_n)

    # Low-credibility flags: surfaced, not deleted, per the spec
    low_cred_rows = subset[subset["credibility_score"] < 40]
    flags = [
        {"review_id": r["review_id"], "credibility_score": r["credibility_score"],
         "excerpt": r["clean_text"][:120]}
        for _, r in low_cred_rows.iterrows()
    ]

    return {
        "overall_summary": " ".join(overall),
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
