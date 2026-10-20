# 07 — Experimental Results: graphs, tables, actual test results

| | |
|---|---|
| **Owner** | Member B |
| **Phase** | 2 |
| **Branch** | `feature/experimental-results` |
| **Depends on** | 02, 03, 04 (Member A) and 05, 06 merged, with their `results/*.json` committed |
| **Dashboard page** | `pages/08_Experimental_Results.py` |
| **Outputs** | `docs/figures/*.png`, `docs/RESULTS.md`, `results/tables/*.csv` |

## Goal

Pull every experiment into **one place** that serves as (a) the
"Experimental Results" dashboard and (b) the source of the report's figures
and tables. Nothing new is measured here. The work is aggregation,
consistency and export.

Also add a **single command that reproduces everything**, so the numbers in
the report can be regenerated and checked.

## Files to add

| File | Purpose |
|---|---|
| `experiments/run_all.py` | Runs, in order: `run_dataset_profile` → `run_model_comparison` → `run_cross_validation` → `run_significance` → `run_ablation` → `run_explainability_eval` → `run_summary_comparison` → `export_tables` → `export_figures`. Flags: `--quick` (passes `--quick` through), `--skip <name>`. Prints a timing summary. BERT stays on Colab: the script uses the existing `results/predictions/distilbert.csv` and warns if it's missing. |
| `src/results_summary.py` | `load_all_results() -> dict[name, result]`, `headline_numbers() -> dict` (best model, current RF F1, contamination reduction, top ablation feature, SHAP/z-score agreement), `master_tables() -> dict[name, DataFrame]` |
| `experiments/export_tables.py` | Writes every master table to `results/tables/<name>.csv` **and** `docs/RESULTS.md` (Markdown tables + metadata: date, git commit, rows). Also writes `.tex` via `DataFrame.to_latex()`, in case the report is in LaTeX. |
| `experiments/export_figures.py` | Imports the chart builders from `src/charts/*` (written in 01–06), renders each one with the committed results, and saves `docs/figures/<NN>_<name>.png` via `fig.write_image(..., scale=2)` |
| `pages/08_Experimental_Results.py` | Dashboard (below) |
| `tests/test_results_summary.py` | `headline_numbers()` works on fixture JSONs in `tests/fixtures/results/` and handles missing files gracefully |
| `tests/fixtures/results/*.json` | Tiny fake versions of each results file, used only for tests |

## Files to change

| File | Change |
|---|---|
| `src/results_io.py` | Add `list_results() -> list[meta]` for the reproducibility panel |
| `requirements.txt` | `kaleido` (Plotly static image export; pin `kaleido==0.2.1` if the newest version needs Chrome and that causes trouble on Windows) |
| `README.md` | "Reproducing the results" section: `python experiments/run_all.py`, plus the Colab step for BERT |

## Master tables (each becomes CSV + Markdown + LaTeX)

| Table | Rows × Columns | Source |
|---|---|---|
| T1 Dataset | stats | `dataset_profile.json` |
| T2 Model comparison | models × acc/prec/rec/F1/AUC/train-s | `model_comparison.json` |
| T3 Cross-validation | classical models × metric mean ± std | `cv_results.json` |
| T4 Significance | model vs RF × χ², p | `significance.json` |
| T5 Ablation (LOO) | feature removed × F1 mean ± std, ΔF1 | `ablation.json` |
| T6 Explanation agreement | method pair × top-1, top-3 Jaccard, τ | `explainability_eval.json` |
| T7 Summarization | variant × contamination, cred., ROUGE-1/2/L, redundancy, coverage | `summary_comparison.json` |

## Figures (`docs/figures/`)

| # | Figure | Builder |
|---|---|---|
| F1 | Class balance & per-category distribution | `charts.dataset` |
| F2 | Model leaderboard (F1) | `charts.model_comparison.leaderboard_bar` |
| F3 | Cost vs quality scatter | `charts.model_comparison.cost_vs_quality_scatter` |
| F4 | Confusion matrices (deployed RF, best model) | `charts.performance.confusion_matrix_heatmap` |
| F5 | ROC curves, all models | `charts.performance.roc_curves` |
| F6 | Ablation ΔF1 | `charts.ablation.delta_bar` |
| F7 | Global SHAP importance | `charts.explainability.shap_bar` |
| F8 | SHAP waterfall for an example fake review | `charts.explainability.shap_waterfall` |
| F9 | Contamination per category (standard vs ours) | `charts.summarization.contamination_bar` |
| F10 | ROUGE comparison | `charts.summarization.rouge_grouped_bar` |

## Dashboard — `pages/08_Experimental_Results.py`

```
Experimental Results                       generated 2026-.. · commit abc1234 · 40,412 reviews

Headline numbers
┌ Best model F1 ┬ Deployed RF F1 ┬ Fake contamination ↓ ┬ Most important feature ┬ SHAP–zscore agreement ┐

[ Tabs: Summary | Tables | Figures | Reproducibility ]

Summary:      4–6 auto-generated "key findings" sentences built from headline_numbers()
              (no hand-typed numbers, so they never go stale)
              + the 3 most important charts (leaderboard, ablation, contamination)
Tables:       selectbox(T1…T7) → st.dataframe + download buttons [CSV] [Markdown] [LaTeX]
Figures:      grid of all figures (live Plotly) + [Download PNG] per figure
              + "Export all to docs/figures" button (runs export_figures in-process)
Reproducibility:
              table of every results file: name, generated_at, git commit, rows, script
              ⚠ warns if results come from different commits or different datasets
              code block with the exact commands (run_all + Colab step)
```

The "different commits" warning matters. If ablation was run on an older
feature set than model comparison, the tables don't belong together. Re-run
before the final report.

## Commits (in order)

1. `feat(results): add results aggregation and headline numbers`
2. `feat(experiments): add run_all orchestrator for reproducible results`
3. `feat(experiments): export master tables to CSV, Markdown and LaTeX`
4. `feat(experiments): export report figures from shared chart builders`
5. `feat(dashboard): add Experimental Results page`
6. `test(results): add aggregation tests with fixture results`
7. `chore(results): regenerate all results and export tables and figures`
8. `docs: add reproducing-results section to README and generated RESULTS.md`

## Done when

- `python experiments/run_all.py` runs end to end (BERT predictions taken from the file).
- `docs/RESULTS.md` and `docs/figures/` are generated, not hand-edited.
- The page shows all 7 tables and 10 figures, plus the reproducibility panel without commit/dataset warnings.

## Report notes

- The report's "Experimental Results" chapter is these tables and figures, in this order: dataset → model comparison → performance/CV/significance → ablation → explainability → summarization.
- Every number in the report should come from `docs/RESULTS.md`. Copy it from there, never retype it.
