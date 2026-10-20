import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLUMNS
from shap_explainer import shap_values, base_value, global_shap_importance, shap_row_factors
from lime_explainer import lime_tabular_explain, lime_text_explain, highlight_html
from explainability import explain_dataframe, explain_review_shap
from explanation_eval import method_agreement, deletion_faithfulness, class_contrast


@pytest.fixture(scope="module")
def trained(kaggle_small):
    from credibility_model import train_model, score_reviews
    clf, _, _ = train_model(kaggle_small)
    return clf, score_reviews(kaggle_small, clf)


def test_shap_values_add_up_to_prediction(trained):
    clf, df = trained
    X = df[FEATURE_COLUMNS].head(50)
    vals = shap_values(clf, X)
    assert vals.shape == (50, len(FEATURE_COLUMNS))
    proba = clf.predict_proba(X)[:, list(clf.classes_).index(1)]
    assert np.allclose(vals.sum(axis=1) + base_value(clf), proba, atol=1e-6)


def test_global_and_row_factors(trained):
    clf, df = trained
    imp = global_shap_importance(clf, df)
    assert list(imp.index) != [] and (imp >= 0).all()
    factors = shap_row_factors(shap_values(clf, df.head(1))[0], df.iloc[0])
    assert len(factors) == len(FEATURE_COLUMNS)
    assert abs(factors[0][2]) >= abs(factors[-1][2])


def test_shap_explanations_keep_output_format(trained):
    clf, df = trained
    out = explain_dataframe(df, method="shap", clf=clf)
    for col in ["positive_factors", "negative_factors", "explanation_text"]:
        assert col in out.columns
    assert out["explanation_text"].str.len().gt(0).all()


def test_shap_explanation_direction_follows_sign():
    row = pd.Series({f: 1.0 for f in FEATURE_COLUMNS})
    means = pd.Series({f: 0.0 for f in FEATURE_COLUMNS})
    contributions = [0.0, 0.2, 0.0, -0.1, 0.0, 0.0]  # length up, repetition down
    out = explain_review_shap(row, contributions, means)
    assert out["positive_factors"] == ["a detailed, longer review"]
    assert out["negative_factors"] == ["repetitive wording"]


def test_lime_tabular_returns_every_feature(trained):
    clf, df = trained
    out = lime_tabular_explain(clf, df.head(300), df.iloc[0], num_samples=300)
    assert sorted(d["feature"] for d in out) == sorted(FEATURE_COLUMNS)


def test_lime_text_and_highlight(kaggle_small):
    from model_zoo import build_text_explainer_model
    model = build_text_explainer_model().fit(kaggle_small["clean_text"], kaggle_small["ground_truth_label"])
    words = lime_text_explain(model, "great value, works well for 3 months", num_samples=200)
    assert all(isinstance(w, str) for w, _ in words)
    html_out = highlight_html("<script>alert(1)</script> great", [("great", 0.5)])
    assert "<script>" not in html_out and "&lt;script&gt;" in html_out
    assert "great</span>" in html_out


def test_agreement_and_faithfulness(trained):
    clf, df = trained
    same = [list(FEATURE_COLUMNS)] * 5
    flipped = [list(reversed(FEATURE_COLUMNS))] * 5
    agree = method_agreement({"zscore": same, "shap": same, "lime": flipped})
    assert agree["top1_match"]["zscore_vs_shap"] == 1.0
    assert agree["top1_match"]["shap_vs_lime"] == 0.0
    faith = deletion_faithfulness(clf, df.head(5), {"shap": same}, df[FEATURE_COLUMNS].mean(), ks=(1, 2))
    assert set(faith["prob_change"]) == {"shap", "random"}
    contrast = class_contrast(np.zeros((4, 6)), pd.Series([0, 1, 0, 1]))
    assert set(contrast) == {"genuine", "fake"}
