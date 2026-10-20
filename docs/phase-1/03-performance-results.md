# 03 — Performance Results: accuracy, precision, recall, F1, confusion matrix

| | |
|---|---|
| **Owner** | Member A |
| **Phase** | 1 |
| **Branch** | `feature/performance-results` |
| **Depends on** | 02 Model Comparison (predictions files) |
| **Dashboard page** | `pages/04_Performance.py` |
| **Results files** | `results/cv_results.json`, `results/significance.json` |

## Goal

02 answers "which model is best?". 03 answers "**how well does each model
work, where does it fail, and are the numbers reliable?**":

- The full metric set per model: accuracy, precision, recall, F1 (per class plus macro), ROC-AUC, PR-AUC, and the confusion matrix
- **5-fold stratified cross-validation** (mean ± std), so results don't rest on one lucky split
- Per-category performance (10 Amazon categories)
- Decision-threshold analysis (today the code hard-codes ≥70 = High and <40 = Low)
- Statistical significance (McNemar's test) between the current RF and the best model

This page **replaces** the "📊 Model Performance Evaluation" expander in the
Review Explorer.

## Files to add

| File | Purpose |
|---|---|
| `src/performance.py` | `per_class_report(y_true, y_pred)`, `per_category_metrics(pred_df, df)`, `threshold_sweep(y_true, score, steps=101)`, `mcnemar(y_true, pred_a, pred_b)` (scipy: exact binomial when there are fewer than 25 disagreements, otherwise chi² with continuity correction) |
| `experiments/run_cross_validation.py` | 5-fold `StratifiedKFold(shuffle=True, random_state=42)` for the classical models in `model_zoo` (BERT excluded, too expensive; say so on the page) → `results/cv_results.json` |
| `experiments/run_significance.py` | Reads `results/predictions/*.csv`, runs McNemar `rf_handcrafted` vs every other model → `results/significance.json` |
| `src/charts/performance.py` | `confusion_matrix_heatmap(cm, normalize)`, `roc_curves(models)`, `pr_curves(models)`, `threshold_curve(sweep)`, `per_category_bar(df)`, `cv_box(cv_results)` |
| `pages/04_Performance.py` | Dashboard (below) |
| `tests/test_performance.py` | `threshold_sweep` is monotonic in recall; McNemar on identical predictions gives p = 1 |

## Files to change

| File | Change |
|---|---|
| `pages/10_Review_Explorer.py` | Remove the evaluation expander and replace it with `st.page_link("pages/04_Performance.py", label="See full performance results →")` |
| `src/credibility_model.py` | Move the High/Medium/Low cut-offs (70 / 40) into module constants `HIGH_THRESHOLD`, `LOW_THRESHOLD`, so the page can show where they sit on the threshold curve. **Only** `score_reviews` and `summarizer.py` read them. |
| `src/summarizer.py` | Replace the literal `70` / `40` with the constants from `credibility_model` |
| `requirements.txt` | nothing (McNemar is implemented with scipy) |

## `results/cv_results.json` → `data`

```json
{
  "k": 5,
  "models": {
    "rf_handcrafted": {"accuracy": [0.75, 0.76, 0.75, 0.75, 0.76], "f1": [], "precision": [], "recall": [], "roc_auc": []},
    "lr_tfidf": {}
  }
}
```

## Dashboard — `pages/04_Performance.py`

```
Performance Results
[ Model ▾ rf_handcrafted ]   [ ☐ Compare with second model ▾ ]

┌ Accuracy ┬ Precision ┬ Recall ┬ F1 ┬ ROC-AUC ┬ PR-AUC ┐   ← st.metric, delta vs current RF
└──────────┴───────────┴────────┴────┴─────────┴────────┘

[ Tabs: Confusion Matrix | Curves | Threshold | By Category | Cross-Validation | Significance ]

Confusion Matrix:  px.imshow 2×2, annotated counts, toggle "normalize by row"
                   + per-class precision/recall/F1 table (fake vs genuine)
                   + 3 example false positives / false negatives (review text)
Curves:            ROC (all models overlaid, AUC in legend) | PR curves
Threshold:         slider 0–1 on the selected model's score → live confusion matrix
                   and metrics; vertical lines at 0.40 and 0.70 (current Low/High cut-offs)
By Category:       bar of F1 per category; flags worst category
Cross-Validation:  box plot of 5 fold scores per model (mean ± std table beside it)
Significance:      table: model vs rf_handcrafted, χ², p-value, "significant at 0.05?"
```

All of this comes from `results/predictions/*.csv`, `cv_results.json` and
`significance.json`. The page needs no retraining and loads instantly. If a
file is missing, `show_result_or_hint()` prints the command to generate it.

The false-positive/false-negative examples join prediction rows to text through
`review_id`, using the live full dataset from `require_pipeline()`. Show them
only when Home's data source is `kaggle_full`.

## Commits (in order)

1. `refactor(model): move credibility thresholds into named constants`
2. `feat(eval): add per-class, per-category, threshold sweep and McNemar helpers`
3. `feat(experiments): add 5-fold cross-validation script`
4. `feat(experiments): add McNemar significance script`
5. `feat(charts): add confusion matrix, ROC/PR, threshold and CV charts`
6. `feat(dashboard): add Performance Results page`
7. `refactor(dashboard): link Review Explorer to Performance page instead of inline metrics`
8. `test(eval): add performance helper tests`
9. `chore(results): add cross-validation and significance results`

## Done when

- Accuracy, precision, recall, F1 and the confusion matrix are shown for every model in 02.
- CV mean ± std is available for every classical model.
- The threshold slider updates the confusion matrix live.
- The Review Explorer no longer computes metrics itself.

## Report notes

- Main results table: mean ± std from 5-fold CV for classical models. Single-split numbers for BERT, clearly marked.
- One confusion-matrix figure for the deployed RF and one for the best model.
- Interpret the errors. The current RF has high recall (0.836) but lower precision (0.718) for "genuine": it calls some fake reviews genuine. Explain what that means for the summaries (some fake text gets moderate weight).
- Report the McNemar p-values whenever you claim one model is "better".
