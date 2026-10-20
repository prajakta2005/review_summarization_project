# 04 — Ablation Study: which features matter?

| | |
|---|---|
| **Owner** | Member A |
| **Phase** | 1 |
| **Branch** | `feature/ablation-study` |
| **Depends on** | 01 Dataset (02 recommended, for the `hybrid` representation) |
| **Dashboard page** | `pages/07_Ablation_Study.py` |
| **Results file** | `results/ablation.json` |

## Goal

Remove features one at a time, retrain, and measure how much performance
drops. That is direct evidence of which credibility signals matter, which
the RF's `feature_importances_` (the current "global importance" chart)
can't give on its own: impurity importance is biased toward
high-cardinality features like `length`.

Four experiments, each run with **5-fold stratified CV** (mean ± std):

| Experiment | What is varied | Question it answers |
|---|---|---|
| **A. Leave-one-out** | Full 6 features minus one, for each of the 6 | How much does losing this feature hurt? |
| **B. Single-feature** | Each feature alone | How much signal does this feature carry on its own? |
| **C. Forward addition** | Add features greedily, best first | How many features do we really need? |
| **D. Group ablation** | Remove groups: *style* (`exclaim_density`, `repetition_ratio`), *content* (`length`, `specificity`), *sentiment* (`sentiment`, `rating_consistency`) | Which kind of signal matters most? |

Optional **E. Representation ablation** (cheap if 02 is merged): `lr_hybrid`
vs `lr_tfidf` shows what the 6 handcrafted features add on top of text.

Run A–D for **two models**, RF (deployed) and LR (linear). Agreement between
them makes the conclusion stronger.

Cross-check: compare the ablation ranking with (i) RF impurity importance,
(ii) **permutation importance** (`sklearn.inspection.permutation_importance`
on the test split), and (iii) mean |SHAP| from feature 05, if
`results/explainability_eval.json` exists. Report Spearman rank correlation
between the rankings.

## Files to add

| File | Purpose |
|---|---|
| `src/ablation.py` | `cv_score(df, features, model_factory, k=5) -> dict[metric, list]`, `leave_one_out(...)`, `single_feature(...)`, `forward_selection(...)`, `group_ablation(groups, ...)`, `importance_rankings(clf, X_test, y_test)` |
| `experiments/run_ablation.py` | Runs A–D (+E if `model_comparison.json` exists) on the full dataset for RF and LR → `results/ablation.json`. CLI flag `--quick` uses 3 folds on a 10k stratified sample for development. |
| `src/charts/ablation.py` | `delta_bar(loo)` (ΔF1 with error bars), `single_feature_bar`, `forward_curve`, `group_bar`, `ranking_comparison_table` |
| `pages/07_Ablation_Study.py` | Dashboard (below) |
| `tests/test_ablation.py` | On the sample df: leave-one-out returns 6 entries, and each uses 5 features |

## Files to change

| File | Change |
|---|---|
| `src/features.py` | Add `FEATURE_GROUPS = {"style": [...], "content": [...], "sentiment": [...]}` next to `FEATURE_COLUMNS` |
| `src/explainability.py` | Add `permutation_feature_importance(clf, X, y)`, so the Review Explorer can offer it as an alternative to impurity importance |
| `app.py` (Home) | Under the global importance chart, add a caption with a `st.page_link` to the Ablation page: "Impurity importance can be biased — see ablation" |

## `results/ablation.json` → `data`

```json
{
  "k": 5, "metric_primary": "f1",
  "models": {
    "rf": {
      "full": {"f1": {"mean": 0.772, "std": 0.004}, "accuracy": {}, "roc_auc": {}},
      "leave_one_out": [{"removed": "specificity", "f1": {"mean": 0.0, "std": 0.0}, "delta_f1": -0.0}],
      "single_feature": [{"feature": "length", "f1": {"mean": 0.0, "std": 0.0}}],
      "forward": [{"step": 1, "added": "length", "features": ["length"], "f1": {"mean": 0.0, "std": 0.0}}],
      "groups": [{"removed_group": "style", "features": ["exclaim_density", "repetition_ratio"], "delta_f1": 0.0}]
    },
    "lr": {}
  },
  "rankings": {
    "ablation_loo": ["..."], "impurity": ["..."], "permutation": ["..."], "shap": ["..."],
    "spearman_vs_ablation": {"impurity": 0.0, "permutation": 0.0, "shap": 0.0}
  },
  "representation": {"lr_tfidf_f1": 0.0, "lr_hybrid_f1": 0.0}
}
```

## Dashboard — `pages/07_Ablation_Study.py`

```
Ablation Study                                   5-fold CV on 40,412 reviews
[ Model: (•) Random Forest ( ) Logistic Regression ]  [ Metric: F1 ▾ ]

┌ Full model F1 ┬ Most important (LOO) ┬ Least important ┬ Features needed for 99% of F1 ┐

[ Tabs: Leave-one-out | Single feature | Forward addition | Feature groups | Rankings ]

Leave-one-out:     horizontal bar of ΔF1 when removed (sorted, red = big drop),
                   error bars = fold std; dashed line at 0
Single feature:    bar of F1 using only that feature; dashed line at 0.5 (chance)
Forward addition:  line chart F1 vs #features, points labelled with added feature,
                   horizontal band = full-model F1 ± std
Feature groups:    bar of ΔF1 per removed group + which features are in each
Rankings:          table: feature × (ablation, impurity, permutation, SHAP rank)
                   + Spearman ρ metrics; highlight disagreements
```

Under each chart, add one auto-generated sentence built from the JSON, e.g.
"Removing **specificity** costs the most F1 (−0.041 ± 0.003)". The report
can then quote the page directly.

## Commits (in order)

1. `feat(features): define feature groups for ablation`
2. `feat(ablation): add CV-based leave-one-out, single-feature and forward selection`
3. `feat(ablation): add group ablation and importance ranking comparison`
4. `feat(explain): add permutation importance helper`
5. `feat(experiments): add ablation study script with quick mode`
6. `feat(charts): add ablation chart builders`
7. `feat(dashboard): add Ablation Study page`
8. `feat(dashboard): link global importance on Home to ablation results`
9. `test(ablation): add ablation unit tests`
10. `chore(results): add full-dataset ablation results`

## Done when

- `python experiments/run_ablation.py` finishes on the full dataset (expected to take a few minutes: about 6+6+21+3 RF fits × 5 folds), and `--quick` finishes in about 1.5 minutes.
- The page shows all tabs for RF and LR.
- The Rankings tab shows SHAP once 05 is merged, and says "SHAP pending" before that.

## Report notes

- Main ablation table: feature removed → F1 (mean ± std) → ΔF1. Do this for RF, with LR in an appendix.
- Say explicitly whether impurity importance agreed with ablation. A disagreement is a finding worth reporting.
- Tie it back to explainability. Features that matter in ablation should also show up most in the per-review explanations (05). If they don't, discuss why.
