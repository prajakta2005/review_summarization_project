import numpy as np
import pandas as pd
import pytest

from summarizer import standard_summary, oracle_summary, credibility_aware_summary, rank_sentences
from summary_eval import contamination_rate, evaluate_summaries, redundancy, attribute_coverage


def make_df():
    texts = {
        1: ("Battery lasts 3 days with heavy use. The strap broke after 2 weeks.", 1, 90.0),
        2: ("Charging takes 2 hours which is fine. Sound quality is clear at full volume.", 1, 85.0),
        3: ("Great product great price. Love it so much, great great great.", 0, 15.0),
        4: ("Amazing product, amazing quality. Best purchase ever, highly recommend.", 0, 20.0),
        5: ("The price is fair for the build quality. Shipping arrived in 4 days.", 1, 75.0),
        6: ("Great quality great value. Great product, buy it.", 0, 10.0),
    }
    return pd.DataFrame([{"review_id": rid, "product_id": "P", "clean_text": t, "ground_truth_label": lab,
                          "credibility_score": score} for rid, (t, lab, score) in texts.items()])


def test_oracle_uses_only_genuine_reviews():
    out = oracle_summary(make_df(), "P", top_n=3)
    assert out["items"] and all(not it["is_fake"] for it in out["items"])
    assert contamination_rate(out["items"]) == 0.0


def test_standard_summary_ignores_scores():
    df = make_df()
    shuffled = df.assign(credibility_score=np.random.default_rng(0).permutation(df["credibility_score"]))
    a = standard_summary(df, "P", 3)["summary"]
    b = standard_summary(shuffled, "P", 3)["summary"]
    assert a == b


def test_credibility_weighting_reduces_contamination():
    df = make_df()
    std = contamination_rate(standard_summary(df, "P", 3)["items"])
    ours = credibility_aware_summary(df, "P", 3)
    assert contamination_rate(ours["overall_items"]) <= std
    assert ours["sources"] == [it["review_id"] for it in ours["overall_items"]]


def test_hard_threshold_excludes_low_scores():
    out = credibility_aware_summary(make_df(), "P", 3, hard_threshold=50)
    assert all(it["score"] >= 50 for it in out["overall_items"])


def test_rank_sentences_indices_match_sentences():
    sents = ["alpha beta", "beta gamma", "delta epsilon", "beta beta alpha"]
    idx = rank_sentences(sents, [1, 1, 1, 1], 2, return_indices=True)
    assert rank_sentences(sents, [1, 1, 1, 1], 2) == [sents[i] for i in idx]


def test_evaluate_summaries_metrics():
    out = evaluate_summaries(make_df(), "P", top_n=3)
    assert set(out["metrics"]) == {"standard", "credibility_aware", "oracle"}
    assert out["metrics"]["oracle"]["rouge1"] == 1.0  # oracle vs itself
    assert redundancy(["same words here", "same words here"]) == pytest.approx(1.0)
    assert attribute_coverage("the battery is great", ["battery", "price"]) == 0.5
