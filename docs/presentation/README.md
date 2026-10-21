# Presentation & Demo Guide

Everything the two presenters need for the second review: what each feature does, the result
to quote, which chart to show, and exactly what to click in the live demo.

All numbers come from [`docs/RESULTS.md`](../RESULTS.md) (generated from the experiments on
the full 40,412-review dataset). If you re-run `python experiments/run_all.py`, re-run
`python experiments/export_paper_figures.py` too, so the charts match.

## Files

| File | Use it for |
|---|---|
| [speaker-notes.md](speaker-notes.md) | Slide-by-slide script: who presents, opening line, points, hand-offs, timing |
| [demo-runbook.md](demo-runbook.md) | Setup, pre-demo checklist, the 6-minute live demo script, what to do if something breaks |
| [01-dataset.md](01-dataset.md) | Full dataset (40k reviews) |
| [02-model-comparison.md](02-model-comparison.md) | RF vs LR vs SVM vs BERT |
| [03-performance.md](03-performance.md) | Accuracy, precision, recall, F1, confusion matrix |
| [04-explainability.md](04-explainability.md) | SHAP & LIME |
| [05-summarization.md](05-summarization.md) | Standard vs credibility-aware summaries (**headline result**) |
| [06-ablation.md](06-ablation.md) | Which features matter |
| [07-experimental-results.md](07-experimental-results.md) | All tables and graphs in one place |
| [08-system-output.md](08-system-output.md) | Live Demo page and screenshots |
| [09-novelty.md](09-novelty.md) | What's new, comparison with existing systems |

## Talk structure (follows the review guidelines)

The guidelines ask for most of the time on **results, comparison, novelty and outcomes**, and
only a short recap of what the first review covered. Suggested 15-minute talk + 5-minute demo:

| Part | Slides | Time | Presenter |
|---|---|---|---|
| Recap of first review (problem, pipeline, objectives) | 2–3 | 1.5 min | A |
| **Results & analysis**: summaries, models, performance, ablation, explainability | 4–10 | 6 min | A (models, ablation) · B (summaries, explainability) |
| **Comparison matrix + comparison with existing systems** | 11–12 | 2 min | B |
| **Novelty** | 13 | 1.5 min | B |
| **Objectives achieved** | 14 | 1.5 min | A |
| Advantages & limitations, future scope | 15–16 | 2 min | A |
| **Live demo** | — | 5 min | B drives, A narrates |
| Conclusion | 17 | 0.5 min | both |

## Chart index (for slides and the paper)

PNG (16:9, for slides) and SVG (vector, for the paper) are both in [`docs/figures/paper/`](../figures/paper/).

| Chart | Shows | Use in |
|---|---|---|
| `P01_contamination_by_length` | Fake content in summaries: 92% → 42% (5 sentences) | Headline slide · paper Results |
| `P02_contamination_per_category` | Drop in 9 of 10 categories | Results · paper Results |
| `P03_rouge` | ROUGE vs genuine-only reference | Results · paper Results |
| `P04_threshold_tradeoff` | Soft weighting vs hard filtering | Discussion / limitations |
| `P05_model_comparison_cv` | 5-fold CV F1 for 9 models | Results · paper Model comparison |
| `P06_accuracy_vs_explainability` | Accuracy vs cost vs interpretability | Comparison / trade-off |
| `P07_confusion_matrices` | Deployed RF vs best model | Performance |
| `P08_ablation` | ΔF1 when each feature is removed | Ablation |
| `P09_explanation_faithfulness` | Deletion test: SHAP most faithful | Explainability |
| `P10_explanation_agreement` | z-score vs SHAP vs LIME agreement | Explainability |
| `P11_importance_rankings` | Four importance measures agree on the top two | Ablation / discussion |
| `P12_feature_separation` | Which signals separate genuine from fake | Dataset |
| `P13_capability_matrix` | Existing approaches vs this system | Comparison matrix |

The dashboard's own figures (F1–F10) are in [`docs/figures/`](../figures/) and the page
screenshots in [`docs/screenshots/`](../screenshots/) (captions in [SYSTEM_OUTPUT.md](../SYSTEM_OUTPUT.md)).
