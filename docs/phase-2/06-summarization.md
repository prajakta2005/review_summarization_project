# 06 — Summarization: normal vs credibility-aware

| | |
|---|---|
| **Owner** | Member B |
| **Phase** | 2 |
| **Branch** | `feature/summary-comparison` |
| **Depends on** | 00 Foundation (01 for full-dataset runs) |
| **Dashboard page** | `pages/06_Summarization.py` |
| **Results file** | `results/summary_comparison.json` |

## Goal

Show, with numbers, that credibility-aware summaries are better than normal
summaries. That's the project's central claim, so it needs a fair experiment.

**Three summarizers**, using the same TF-IDF centroid ranker
(`rank_sentences`) so the *only* difference is credibility:

| Summary | Sentences from | Weight |
|---|---|---|
| **Standard (baseline)** | all reviews | uniform 1.0 |
| **Credibility-aware (ours)** | all reviews | `credibility_score / 100`, floor 0.05 (current `credibility_aware_summary`) |
| **Oracle (reference)** | only ground-truth **genuine** (`OR`) reviews | uniform 1.0 |

The dataset has real fake/genuine labels, so we can measure directly how much
**fake content gets into each summary**.

### Metrics (per category, then averaged over the 10 categories)

| Metric | Definition | Better |
|---|---|---|
| **Fake contamination rate** | % of summary sentences that come from ground-truth fake (`CG`) reviews | lower |
| **Mean source credibility** | mean `credibility_score` of the reviews behind the summary sentences | higher |
| **ROUGE-1 / 2 / L vs oracle** | overlap with the genuine-only reference summary (`rouge-score`, F-measure) | higher |
| **Redundancy** | mean pairwise TF-IDF cosine between summary sentences | lower |
| **Coverage** | share of the category's top attributes (from `comparison.ATTRIBUTE_KEYWORDS`) mentioned in the summary | higher |

Run the evaluation at `top_n ∈ {3, 5, 8}`. Then run a **sensitivity sweep**
over the weight floor (0.0, 0.05, 0.2, 0.5) and a hard filter variant
("drop reviews < 40"), so we can show *how much* weighting matters.

### Avoid leakage (important)

If the RF is trained on the same reviews it then scores, the credibility scores
are over-optimistic and so is the improvement. Use **out-of-fold scores**:
`cross_val_predict(rf, X, y, cv=5, method="predict_proba")` gives every one of
the 40,412 reviews a score from a model that never saw it. The evaluation
script must use these. The live page may use the normal pipeline scores, but
the page must say so.

### Optional, if there's time

- **LLM judge**: Gemini (already wired up in `llm_explainer.py`) does a blind A/B comparison: "Which summary is more trustworthy and informative?". Randomize the order and use 10 categories × 3 `top_n` settings.
- **Human evaluation**: 3–5 classmates rate 10 blind pairs on a 1–5 scale for trustworthiness and informativeness. Use a Google Form, and store the anonymized ratings in `results/human_eval.csv`.

## Files to add

| File | Purpose |
|---|---|
| `src/summary_eval.py` | `contamination_rate`, `mean_source_credibility`, `rouge_vs_reference`, `redundancy`, `attribute_coverage`, `evaluate_summaries(df, product_id, top_n, weight_floor) -> dict` |
| `experiments/run_summary_comparison.py` | Out-of-fold scores → for every category × `top_n` × variant, generate 3 summaries + metrics → `results/summary_comparison.json`. Saves the example summary texts too, for the page and report. |
| `src/charts/summarization.py` | `contamination_bar(per_category)`, `rouge_grouped_bar`, `metric_delta_table`, `sensitivity_line(weight_floor)`, `topn_line` |
| `pages/06_Summarization.py` | Dashboard (below) |
| `tests/test_summary_eval.py` | Contamination of a summary built only from genuine rows is 0. The standard summary ignores scores (shuffling scores doesn't change it). The oracle uses only label-1 rows. |

## Files to change

| File | Change |
|---|---|
| `src/summarizer.py` | (1) `rank_sentences(..., return_indices=False)`: when True, returns indices so each summary sentence can be traced to its review. (2) Extract `_collect_sentences(subset)` → sentences, review_ids, scores, labels. (3) Add `standard_summary(df, product_id, top_n)` and `oracle_summary(df, product_id, top_n)`. (4) `credibility_aware_summary` gains `weight_floor=0.05` and `hard_threshold=None` params and returns `"sources": [review_id, ...]` for the overall summary. Existing keys stay unchanged. |
| `pages/10_Review_Explorer.py` | The "Credibility-Aware Summary by Product" section gets a small "Compare with standard summary →" `st.page_link` |
| `requirements.txt` | `rouge-score` |

## `results/summary_comparison.json` → `data`

```json
{
  "score_source": "out_of_fold_5cv",
  "settings": {"top_n": [3, 5, 8], "weight_floor": [0.0, 0.05, 0.2, 0.5]},
  "per_category": [
    {"category": "Books_5", "top_n": 5, "variant": "credibility_aware", "weight_floor": 0.05,
     "contamination": 0.0, "mean_source_cred": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0,
     "redundancy": 0.0, "coverage": 0.0}
  ],
  "averages": {
    "standard":          {"contamination": 0.0, "rouge1": 0.0, "...": 0.0},
    "credibility_aware": {"contamination": 0.0, "rouge1": 0.0, "...": 0.0}
  },
  "examples": [{"category": "Books_5", "top_n": 5,
                "standard": [{"sentence": "...", "review_id": 1, "score": 0.0, "is_fake": true}],
                "credibility_aware": [], "oracle": []}]
}
```

## Dashboard — `pages/06_Summarization.py`

```
Summarization: Standard vs Credibility-Aware
[ Category ▾ ]  [ Sentences: 3 · 5 · 8 ]  [ Weight floor ── 0.05 ]  [ ☐ hard filter < 40 ]

┌ Fake contamination ┬ Mean source cred. ┬ ROUGE-L vs genuine-only ┐
│ 18% → 4%  ▼ -14pp  │ 52 → 81  ▲ +29    │ 0.31 → 0.47  ▲ +0.16    │   ← st.metric with delta
└────────────────────┴───────────────────┴─────────────────────────┘

┌ Standard summary ───────────────┐ ┌ Credibility-aware summary ──────┐
│ ● "Love this! Best ever..."  🔴 │ │ ● "Battery lasted 3 weeks..." 🟢│
│   review #812 · cred 31% · FAKE │ │   review #77 · cred 88% · GENUINE│
│ ● ...                           │ │ ● ...                           │
└─────────────────────────────────┘ └─────────────────────────────────┘
  each sentence: coloured left border by credibility, badge = ground-truth label
  (oracle summary in an expander)

[ Tabs: All categories | Sensitivity | Method ]
All categories:  grouped bar contamination per category (standard vs ours)
                 + grouped bar ROUGE-1/2/L + table of averages (from results JSON)
Sensitivity:     line: contamination & ROUGE-L vs weight floor; line vs top_n
Method:          diagram + text explaining the three summarizers and leakage control
```

The top half (selected category) is computed **live** from
`require_pipeline()`, so the sliders react immediately. The "All categories"
tab reads `summary_comparison.json` (out-of-fold, full dataset), which is the
number the report uses.

## Commits (in order)

1. `refactor(summary): extract sentence collection and return source indices from ranker`
2. `feat(summary): add standard and oracle summarizers as baselines`
3. `feat(summary): add weight floor and hard-threshold options to credibility-aware summary`
4. `feat(summary): add summary evaluation metrics (contamination, ROUGE, redundancy, coverage)`
5. `feat(experiments): add summary comparison script with out-of-fold credibility scores`
6. `feat(charts): add summarization comparison chart builders`
7. `feat(dashboard): add Summarization comparison page`
8. `feat(dashboard): link Review Explorer summary to comparison page`
9. `test(summary): add summary evaluation tests`
10. `chore(results): add summary comparison results on full dataset`

## Done when

- For every category, the page shows the standard and credibility-aware summaries side by side, with each sentence traceable to its review, score and true label.
- `results/summary_comparison.json` uses out-of-fold scores and covers all 10 categories × 3 `top_n` values.
- The averages table shows the change in contamination and ROUGE between standard and credibility-aware summaries.

## Report notes

- This is the headline result. Lead with one sentence like "Credibility-aware summarization reduced fake-sentence contamination from X% to Y% (averaged over 10 categories), while improving ROUGE-L against a genuine-only reference from A to B."
- Explain the oracle reference and its limitation: ROUGE against an extractive genuine-only summary is a proxy, not human judgement. Mention the optional LLM/human evaluation if you did it.
- State the leakage control (out-of-fold scores) explicitly. Examiners will ask.
- Show one worked example: the same category, both summaries, with fake sentences marked.
