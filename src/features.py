"""
features.py
-----------
Extracts the signals the Credibility Model and the Explainability
module both depend on. Each function returns ONE named feature so that
later, when we show "why" a review got its score, we can point
directly at these column names.

Teaching note on sentiment: production systems usually use a
pretrained sentiment model (VADER, a fine-tuned transformer, etc).
This sandbox has no internet access to model/lexicon hubs, so I've
implemented a small rule-based lexicon here instead. It's a fallback,
not the "real" approach -- swap in VADER (`nltk.sentiment.vader`) or
a HuggingFace sentiment pipeline once you're running this locally
with internet access. I call this out explicitly so it's not
mistaken for the final design choice.
"""

import re
import numpy as np
import pandas as pd

# --- lightweight fallback sentiment lexicon (swap for VADER/transformer later) ---
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "love", "perfect", "best",
    "happy", "recommend", "comfortable", "quality", "solid", "impressive",
    "fast", "easy", "reliable", "nice", "worth", "satisfied", "smooth"
}
NEGATIVE_WORDS = {
    "bad", "worst", "terrible", "waste", "poor", "broken", "disappointed",
    "slow", "cheap", "failed", "issue", "problem", "defective", "regret",
    "awful", "horrible", "refund", "useless", "faulty", "glitchy"
}


def _words(text: str):
    return re.findall(r"[a-zA-Z']+", text.lower())


def sentiment_score(text: str) -> float:
    """Simple polarity in [-1, 1]: (pos - neg) / total_sentiment_words."""
    tokens = _words(text)
    pos = sum(1 for w in tokens if w in POSITIVE_WORDS)
    neg = sum(1 for w in tokens if w in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def review_length(text: str) -> int:
    return len(_words(text))


def exclamation_density(raw_text: str) -> float:
    """Exclamation marks per 100 characters -- classic hype/fake signal."""
    if not raw_text:
        return 0.0
    count = raw_text.count("!")
    return round(100 * count / max(len(raw_text), 1), 3)


def word_repetition_ratio(text: str) -> float:
    """1 - (unique words / total words). High = repetitive = suspicious."""
    tokens = _words(text)
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    return round(1 - unique_ratio, 3)


def specificity_score(text: str) -> int:
    """Counts concrete, specific-detail indicators: numbers, units of time,
    named comparisons. More specifics -> more likely a genuine first-hand account.
    """
    tokens = _words(text)
    numeric_mentions = len(re.findall(r"\b\d+\b", text))
    time_units = sum(1 for w in tokens if w in {
        "day", "days", "week", "weeks", "month", "months", "hour", "hours",
        "minute", "minutes", "year", "years"
    })
    return numeric_mentions + time_units


def rating_text_consistency(rating: float, sentiment: float) -> float:
    """How well the star rating aligns with text sentiment.
    Rating normalized to [-1, 1] (rating 3 = neutral midpoint on a 1-5 scale),
    then compared to sentiment. 1.0 = perfectly consistent, 0.0 = totally opposed.
    """
    normalized_rating = (rating - 3) / 2  # maps 1..5 -> -1..1
    diff = abs(normalized_rating - sentiment)
    return round(1 - (diff / 2), 3)  # diff ranges 0..2, so this maps to 0..1


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds one column per feature to the dataframe."""
    df = df.copy()
    df["sentiment"] = df["clean_text_lower"].apply(sentiment_score)
    df["length"] = df["clean_text"].apply(review_length)
    df["exclaim_density"] = df["review_text"].apply(exclamation_density)
    df["repetition_ratio"] = df["clean_text_lower"].apply(word_repetition_ratio)
    df["specificity"] = df["clean_text"].apply(specificity_score)
    df["rating_consistency"] = df.apply(
        lambda r: rating_text_consistency(r["rating"], r["sentiment"]), axis=1
    )
    return df


FEATURE_COLUMNS = [
    "sentiment", "length", "exclaim_density",
    "repetition_ratio", "specificity", "rating_consistency"
]


if __name__ == "__main__":
    from preprocessing import load_reviews, preprocess
    df = load_reviews("data/sample_reviews.csv")
    df = preprocess(df)
    df = extract_features(df)
    print(df[["review_id"] + FEATURE_COLUMNS])
