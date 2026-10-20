# 09 — Novelty: what's new in combining credibility, explainability and summarization

| | |
|---|---|
| **Owner** | Member B (Member A reviews the claims against the numbers) |
| **Phase** | 2 |
| **Branch** | `docs/novelty` |
| **Depends on** | 06 merged (the core evidence). Final numbers from 07. |
| **Dashboard page** | `pages/01_Novelty.py` (first in the sidebar, since it's the "why this project" page) |
| **Outputs** | `docs/NOVELTY.md` (report-ready text) |

## Goal

Separately, each piece is well studied: fake-review detection, explainable AI,
and review summarization. The novelty is the **coupling**. The credibility
model's output changes *what goes into the summary*, every summary sentence
can be traced back to a scored and explained review, and we **measure** what
that coupling buys. This feature turns that into (a) a clear written claim
backed by our own numbers and (b) a dashboard page that *demonstrates* it.

## The claims (refine wording, keep them evidence-backed)

| # | Claim | Evidence (from our results) |
|---|---|---|
| C1 | **Credibility-gated summarization.** Summary sentences are weighted by the per-review credibility score. Low-credibility reviews are down-weighted and *surfaced separately*, not silently deleted. | `summary_comparison.json`: contamination standard → ours, ROUGE vs genuine-only reference |
| C2 | **End-to-end traceability.** Every summary sentence links to its source review, that review's score, and the SHAP reasons for the score. | Trace view on the Novelty page; `sources` field from 06 |
| C3 | **Explanations checked for faithfulness, not just displayed.** Three explanation methods are compared, and the deployed one is validated with a deletion test. | `explainability_eval.json` |
| C4 | **A deliberate interpretability trade-off, quantified.** We report how much F1 the 6-feature explainable model gives up against TF-IDF/BERT, and whether the handcrafted signals add value on top of text (hybrid). | `model_comparison.json`, `ablation.json` (representation) |
| C5 | **A contamination metric for evaluating trust-aware summaries,** usable whenever ground-truth fake labels exist. | Definition in 06, results in 07 |

**Be careful with "first".** Don't claim "first system to..." unless a
literature search backs it up. Spend time on Google Scholar with queries like
*"credibility-aware review summarization"*, *"fake review detection
summarization"*, *"trustworthy opinion summarization"*, *"explainable fake
review detection"*. List the closest works you find in the related-work table,
and phrase the novelty relative to them ("unlike X, which filters fake reviews
before summarizing, we ...").

Starting points for the related-work table. Verify each reference before citing it:

| Area | Starting reference | What it lacks vs us |
|---|---|---|
| Fake/deceptive review detection | Ott et al., 2011 (ACL), deceptive opinion spam | No summarization, limited explanation |
| Our dataset (computer-generated fake reviews) | Salminen et al., 2022 (Journal of Retailing and Consumer Services) | Detection only |
| Review summarization | Hu & Liu, 2004 (KDD), mining and summarizing customer reviews | Treats all reviews as trustworthy |
| Explainable ML | Ribeiro et al., 2016 (LIME); Lundberg & Lee, 2017 (SHAP) | Generic methods, not tied to summarization |
| (add the closest combined works you find) | | |

## Files to add

| File | Purpose |
|---|---|
| `docs/NOVELTY.md` | Report-ready text: 1-paragraph novelty statement, the claims table with **actual numbers** (copied from `docs/RESULTS.md`), related-work comparison table, limitations |
| `src/novelty.py` | `capability_matrix() -> DataFrame` (rows: approaches, cols: detects fake / explains / summarizes / credibility-weighted summary / traceable / evaluated contamination) and `evidence_cards() -> list[dict]` (built from `results_summary.headline_numbers()` from 07, falling back to "pending" values) |
| `pages/01_Novelty.py` | Dashboard (below) |
| `tests/test_novelty.py` | `evidence_cards()` doesn't crash when results files are missing |

## Files to change

| File | Change |
|---|---|
| `README.md` | Replace the opening paragraph with the 3-sentence novelty statement and link to `docs/NOVELTY.md` |
| `src/summarizer.py` | Docstring "Credibility-aware behavior (the actual novelty)" updated to reference C1/C2 and the contamination metric. No logic change. |

## Dashboard — `pages/01_Novelty.py`

```
What's new in this system

[ Pipeline diagram — st.graphviz_chart (DOT string, no extra install) ]
  Reviews → Preprocess → Features → Credibility model ─┬─► Explanations (SHAP/LIME)
                                                       └─► weights ─► Summary ◄── traceable links
  (credibility → summary edge highlighted: "this is the novel coupling")

Evidence (live from results/)
┌ Fake contamination ┬ ROUGE-L vs genuine ┬ Explanation faithfulness ┬ Interpretability cost ┐
│ 18% → 4%           │ +0.16              │ SHAP drop 0.21 vs 0.05   │ −0.12 F1 vs BERT      │
└────────────────────┴────────────────────┴──────────────────────────┴───────────────────────┘

Capability matrix (✓/✗ table): Detection-only | Summarization-only | XAI-only | This system

Trace a summary sentence  ← the interactive proof of C2
  [ Category ▾ ] → credibility-aware summary sentences as radio options
  pick one → its source review · score gauge · SHAP waterfall · true label
           → "In the standard summary, this sentence would rank #k"

Limitations (honest): category-level not product-level · extractive only ·
  fake = computer-generated in this dataset · 6-feature model trades accuracy for clarity
```

The "Trace a summary sentence" panel reuses functions from 05
(`shap_row_factors`, `shap_waterfall`) and 06 (`credibility_aware_summary(...)["sources"]`).
No new modeling is needed.

## Commits (in order)

1. `feat(novelty): add capability matrix and evidence cards from results`
2. `feat(dashboard): add Novelty page with pipeline diagram and evidence`
3. `feat(dashboard): add summary-sentence trace view to Novelty page`
4. `test(novelty): handle missing results in evidence cards`
5. `docs: add novelty statement with related-work comparison`
6. `docs: update README introduction with novelty statement`

## Done when

- Every claim in `docs/NOVELTY.md` has a number or a screenshot behind it.
- The Novelty page shows live evidence cards, and the trace view works for every category.
- Member A has checked the numbers in `NOVELTY.md` against `docs/RESULTS.md`.

## Report notes

- Put the novelty statement in the Introduction (contributions list) and the Conclusion.
- The capability matrix fits in Related Work.
- The limitations list belongs in the report too. Examiners trust a project more when it names its own limits.
