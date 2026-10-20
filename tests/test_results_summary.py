import os

import pytest

import results_io
from results_summary import load_all_results, headline_numbers, key_findings, master_tables, to_markdown_table

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "results")


@pytest.fixture
def fixture_results(monkeypatch):
    monkeypatch.setattr(results_io, "RESULTS_DIR", FIXTURES)
    return load_all_results()


@pytest.fixture
def no_results(monkeypatch, tmp_path):
    monkeypatch.setattr(results_io, "RESULTS_DIR", str(tmp_path))
    return load_all_results()


def test_headline_numbers(fixture_results):
    h = headline_numbers(fixture_results)
    assert h["best_f1"] == 0.95 and h["deployed_f1"] == 0.77
    assert h["contamination_standard"] == 0.9 and h["contamination_ours"] == 0.4
    assert h["top_feature"] == "sentiment"
    assert not h["bert_included"]
    findings = key_findings(h)
    assert any("90% to 40%" in f for f in findings)


def test_master_tables(fixture_results):
    tables = master_tables(fixture_results)
    assert set(tables) == {"T1_dataset", "T2_model_comparison", "T3_cross_validation", "T4_significance",
                           "T5_ablation", "T6_explanation_agreement", "T7_summarization"}
    assert len(tables["T5_ablation"]) == 7  # full model + 6 removals
    md = to_markdown_table(tables["T2_model_comparison"])
    assert md.count("\n") == 3  # header, separator, 2 rows


def test_missing_results_are_handled(no_results):
    assert headline_numbers(no_results) == {}
    assert master_tables(no_results) == {}
    assert key_findings({}) == []
    assert results_io.list_results() == []


def test_report_figures_build(fixture_results):
    from report_figures import report_figures
    figs = report_figures(fixture_results)
    assert "F9_contamination" in figs and "F8_shap_waterfall_fake" in figs
