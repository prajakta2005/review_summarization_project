import pytest

from features import FEATURE_COLUMNS
from single_review import score_single_review


@pytest.fixture(scope="module")
def model_and_reference(kaggle_small):
    from credibility_model import train_model, score_reviews
    clf, _, _ = train_model(kaggle_small)
    return clf, score_reviews(kaggle_small, clf)


def test_output_has_everything_the_demo_needs(model_and_reference):
    clf, ref = model_and_reference
    category = ref["product_id"].iloc[0]
    out = score_single_review("Works fine. Battery lasts 2 days.", 4, category, clf, ref)
    for key in ["credibility_score", "credibility_label", "confidence", "shap_factors", "base_value",
                "shap_explanation", "zscore_explanation", "feature_comparison", "summary_weight",
                "category_percentile"]:
        assert key in out
    assert 0 <= out["credibility_score"] <= 100
    assert len(out["shap_factors"]) == len(FEATURE_COLUMNS)
    total = out["base_value"] + sum(c for _, _, c in out["shap_factors"])
    assert total * 100 == pytest.approx(out["credibility_score"], abs=0.1)


def test_repetitive_hype_scores_lower_than_specific_review(model_and_reference):
    clf, ref = model_and_reference
    category = ref["product_id"].iloc[0]
    hype = score_single_review("BEST!!! AMAZING!!! LOVE LOVE LOVE!!! BEST BEST BEST!!!", 5, category, clf, ref)
    specific = score_single_review(
        "Used it for 3 weeks on my commute. The battery lasts about 2 days, which matches the listing, "
        "but the strap broke after 10 days and I had to glue it. The sound is clear at half volume "
        "and the case fits in a jacket pocket. Decent for the price, not great.", 3, category, clf, ref)
    assert hype["credibility_score"] < specific["credibility_score"]
