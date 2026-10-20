"""
comparison.py
--------------
Module 9: Review Comparison & Aggregation.
Looks ACROSS reviews for one product to find frequently mentioned
attributes, consensus claims, and conflicting claims.
"""

import re
from collections import Counter
import pandas as pd

from features import POSITIVE_WORDS, NEGATIVE_WORDS

ATTRIBUTE_KEYWORDS = {
    "battery": ["battery", "charge", "charging"],
    "quality": ["quality", "build", "sturdy", "material"],
    "price": ["price", "value", "worth", "expensive", "cheap"],
    "shipping": ["shipping", "delivery", "arrived", "package"],
    "comfort": ["comfortable", "comfort", "fit", "fits"],
    "durability": ["broke", "broken", "durable", "lasted", "wear"],
    "customer_service": ["support", "refund", "customer service", "replace"],
}


def _words(text: str):
    return re.findall(r"[a-zA-Z']+", str(text).lower())


def find_attribute_mentions(text: str) -> list:
    text_lower = str(text).lower()
    return [attr for attr, kws in ATTRIBUTE_KEYWORDS.items() if any(kw in text_lower for kw in kws)]


def sentence_sentiment(text: str) -> str:
    tokens = _words(text)
    pos = sum(1 for w in tokens if w in POSITIVE_WORDS)
    neg = sum(1 for w in tokens if w in NEGATIVE_WORDS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def compare_products(df: pd.DataFrame, product_id: str) -> dict:
    subset = df[df["product_id"] == product_id]
    if subset.empty:
        return {"attribute_sentiment": {}, "consensus": [], "conflicting": [], "credibility_distribution": {}}

    attribute_sentiments = {attr: [] for attr in ATTRIBUTE_KEYWORDS}
    for _, row in subset.iterrows():
        mentioned = find_attribute_mentions(row["clean_text"])
        review_sentiment = sentence_sentiment(row["clean_text"])
        for attr in mentioned:
            attribute_sentiments[attr].append(review_sentiment)

    consensus, conflicting, attribute_summary = [], [], {}
    for attr, sentiments in attribute_sentiments.items():
        if not sentiments:
            continue
        counts = Counter(sentiments)
        attribute_summary[attr] = dict(counts)
        pos, neg = counts.get("positive", 0), counts.get("negative", 0)
        total = pos + neg
        if total == 0:
            continue
        if pos > 0 and neg > 0 and min(pos, neg) / total >= 0.25:
            conflicting.append({"attribute": attr, "positive": pos, "negative": neg})
        else:
            majority = "positive" if pos >= neg else "negative"
            consensus.append({"attribute": attr, "sentiment": majority, "mentions": total})

    return {
        "attribute_sentiment": attribute_summary,
        "consensus": consensus,
        "conflicting": conflicting,
        "credibility_distribution": subset["credibility_label"].value_counts().to_dict(),
        "total_reviews": len(subset),
    }