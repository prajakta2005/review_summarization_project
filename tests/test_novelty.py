import pandas as pd

import results_io
from novelty import capability_matrix, evidence_cards, standard_rank


def test_evidence_cards_pending_without_results(monkeypatch, tmp_path):
    monkeypatch.setattr(results_io, "RESULTS_DIR", str(tmp_path))
    cards = evidence_cards()
    assert len(cards) == 4
    assert all(c["value"] == "pending" for c in cards)


def test_capability_matrix_shape():
    m = capability_matrix()
    assert "This system" in m.index
    assert (m.loc["This system"] == "✓").all()


def test_standard_rank_finds_sentence():
    df = pd.DataFrame({
        "review_id": [1, 2, 3], "product_id": ["P"] * 3, "credibility_score": [80.0, 20.0, 60.0],
        "clean_text": ["Battery life is great. Screen is dim.", "Great great battery.", "Shipping was slow."],
    })
    rank, n = standard_rank(df, "P", 1, "Battery life is great.")
    assert n == 4 and 1 <= rank <= 4
    assert standard_rank(df, "P", 99, "missing")[0] is None
