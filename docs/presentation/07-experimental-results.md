# 7. Experimental Results: graphs, tables, actual test results

**Page:** Experimental Results · **Owner:** Member B

## What it is

One page that gathers every experiment: headline numbers, key findings generated from the
results, seven report tables (T1–T7, downloadable as CSV / Markdown / LaTeX), all figures, and a
**reproducibility panel** showing when, at which commit and on which data each result was produced.

One command regenerates everything (~17 minutes on a laptop):

```bash
python experiments/run_all.py
python experiments/export_paper_figures.py   # the charts in this folder
```

## The tables

| Table | Content |
|---|---|
| T1 | Dataset statistics |
| T2 | Model comparison on the shared test split |
| T3 | 5-fold cross-validation, mean ± std |
| T4 | McNemar significance between every pair of models |
| T5 | Ablation (leave-one-out) |
| T6 | Agreement between explanation methods |
| T7 | Summarization: standard vs credibility-aware, 3 / 5 / 8 sentences |

All in [`docs/RESULTS.md`](../RESULTS.md) and `results/tables/` (`.csv`, `.tex`).

![Contamination](../figures/paper/P01_contamination_by_length.png)

## How to demo

1. **Experimental Results** page → the five headline cards.
2. *Summary* tab → the key findings (written from the data, never typed by hand).
3. *Tables* tab → pick T7, click **⬇ LaTeX** ("this goes straight into the paper").
4. *Reproducibility* tab → "all results from one commit, one dataset" (green box).

**Say:** "Nothing in this talk is hand-typed: every number comes from these result files, and one
command regenerates them."
