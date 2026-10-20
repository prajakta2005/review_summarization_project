# 02 — Model Comparison: Random Forest vs Logistic Regression vs SVM vs BERT

| | |
|---|---|
| **Owner** | Member A |
| **Phase** | 1 |
| **Branch** | `feature/model-comparison` |
| **Depends on** | 01 Dataset (needs `results/split_indices.json`) |
| **Dashboard page** | `pages/03_Model_Comparison.py` |
| **Results files** | `results/model_comparison.json`, `results/predictions/<model_id>.csv` |

## Goal

Train every model on the **same training rows** and test on the **same test
rows** (from `src/splits.py`), then compare quality *and* cost.

A model alone doesn't make the comparison fair. What it learns from matters
too, so compare across **feature sets**:

| Feature set | What it is | Why |
|---|---|---|
| `handcrafted` | The 6 interpretable features in `features.FEATURE_COLUMNS` | What the system uses today. Fully explainable. |
| `tfidf` | Word 1–2-gram TF-IDF of `clean_text` (`max_features=50_000`, `min_df=2`, `sublinear_tf=True`) | Strong classical text baseline |
| `hybrid` | `tfidf` ⊕ scaled `handcrafted` (`scipy.sparse.hstack`) | Tests whether the interpretable features add anything on top of text |
| `bert` | Raw text into a fine-tuned DistilBERT | State-of-the-art reference |

The model × feature-set grid gives 9 classical runs and 1 BERT run:

| model_id | Model | Feature set |
|---|---|---|
| `rf_handcrafted` | `RandomForestClassifier(n_estimators=200, max_depth=4, class_weight="balanced")` (current) | handcrafted |
| `rf_handcrafted_tuned` | RF with `max_depth=None, min_samples_leaf=5` | handcrafted |
| `lr_handcrafted` | `StandardScaler` → `LogisticRegression(max_iter=1000)` | handcrafted |
| `svm_handcrafted` | `StandardScaler` → `SVC(kernel="rbf", probability=False)` (use `decision_function` for AUC) | handcrafted |
| `rf_tfidf` | RF (`n_estimators=300, n_jobs=-1`) | tfidf |
| `lr_tfidf` | `LogisticRegression(C=4, max_iter=2000)` | tfidf |
| `svm_tfidf` | `LinearSVC(C=0.5)` (RBF on 50k-dim sparse text is too slow) | tfidf |
| `lr_hybrid` | `LogisticRegression` | hybrid |
| `svm_hybrid` | `LinearSVC` | hybrid |
| `distilbert` | `distilbert-base-uncased`, fine-tuned 2 epochs | bert |

RBF-SVC on 32k × 6 handcrafted features takes a few minutes. If it's too slow
on a laptop, switch to `LinearSVC` and note that in the report.

## BERT plan (no GPU on our machines)

Fine-tuning DistilBERT on 32k reviews takes hours on CPU and about 15–25 min on
a free **Google Colab T4 GPU**. The plan:

1. Use `results/split_indices.json` (from 01), so Colab uses the identical split.
2. `notebooks/bert_finetune_colab.ipynb`: upload the CSV + split file, then tokenize (`max_length=256`), use `Trainer` with `lr=2e-5, batch=32, epochs=2, weight_decay=0.01, seed=42`, and predict on the test rows.
3. Download **only** `distilbert.csv` (`review_id, y_true, y_pred, score`, the same format as the classical models) and `distilbert_meta.json` (train time, inference time, epochs, model name) into `results/predictions/`.
4. `run_model_comparison.py` merges the BERT predictions into the same metrics computation as the other models.
5. Optional: save the fine-tuned weights to Google Drive (about 260 MB, **never** in git). The Live Demo (08) uses the RF, not BERT.

Fallback if Colab is unavailable: run `--bert-subset 4000` on CPU (train on
4k and test on the full test split). The report must then say "DistilBERT
trained on a 4k subset".

## Files to add

| File | Purpose |
|---|---|
| `src/text_features.py` | `build_tfidf()`, `build_hybrid_matrix(df, tfidf, scaler, fit: bool)` |
| `src/model_zoo.py` | `MODEL_SPECS: dict[model_id, (feature_set, factory)]` and `fit_and_predict(model_id, df, train_idx, test_idx) -> dict(y_true, y_pred, score, train_seconds, predict_seconds)` |
| `experiments/run_model_comparison.py` | Loops `MODEL_SPECS` (optionally `--only lr_tfidf,svm_tfidf`), merges BERT predictions if present, computes metrics with `evaluation.compute_metrics`, writes `results/model_comparison.json` and `results/predictions/<model_id>.csv`. Also saves `models/tfidf_lr.joblib` (a `Pipeline(tfidf, lr)`) for LIME text explanations in 05. |
| `notebooks/bert_finetune_colab.ipynb` | Colab notebook described above |
| `requirements-bert.txt` | `torch`, `transformers`, `datasets`, `accelerate`, `scikit-learn`, `pandas` (Colab only) |
| `src/charts/model_comparison.py` | `leaderboard_bar(metric)`, `metric_heatmap()`, `cost_vs_quality_scatter()`, `featureset_grouped_bar()` |
| `pages/03_Model_Comparison.py` | Dashboard (below) |
| `tests/test_model_zoo.py` | Every non-BERT spec fits and predicts on the sample df; output arrays have test length |

## Files to change

| File | Change |
|---|---|
| `src/evaluation.py` | Add `compute_metrics(y_true, y_pred, score) -> dict` (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC and PR curve points downsampled to ≤200). Make `evaluate_model()` call it. Feature 03 reuses it. |
| `src/credibility_model.py` | No logic change. Add a docstring note that the RF is `rf_handcrafted` in the comparison. |
| `.gitignore` | `models/tfidf_lr.joblib`, `models/bert/` |
| `requirements.txt` | `scipy` (explicit, for sparse hstack) |

## `results/model_comparison.json` → `data`

```json
{
  "models": [
    {
      "model_id": "lr_tfidf", "model": "Logistic Regression", "feature_set": "tfidf",
      "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
      "roc_auc": 0.0, "pr_auc": 0.0,
      "confusion_matrix": [[0, 0], [0, 0]],
      "train_seconds": 0.0, "predict_seconds": 0.0, "n_train": 32329, "n_test": 8083,
      "interpretable": false
    }
  ],
  "best_by_f1": "...",
  "n_test": 8083
}
```

`results/predictions/<model_id>.csv` holds `review_id, y_true, y_pred, score`.
Feature 03 uses these files for ROC curves, threshold tuning and significance
tests, without retraining.

## Dashboard — `pages/03_Model_Comparison.py`

```
Model Comparison                        Test set: 8,083 reviews · same split for all models
[ Filter: feature sets ☑handcrafted ☑tfidf ☑hybrid ☑bert ]  [ Metric: F1 ▾ ]

┌ 🏆 Best: DistilBERT F1 0.9x ┬ Best classical: LR-tfidf 0.8x ┬ Current RF: 0.772 ┐

[ Leaderboard bar: models sorted by selected metric, coloured by feature set ]

[ Heatmap: rows = models, cols = accuracy/precision/recall/F1/AUC, values annotated ]

[ Scatter: x = train time (log), y = F1, size = predict time, colour = interpretable? ]
   → shows the accuracy-vs-explainability-vs-cost trade-off

[ Grouped bar: model family × feature set (RF/LR/SVM over handcrafted/tfidf/hybrid) ]

[ Expander: full table (st.dataframe) + CSV download ]
[ Expander: "Run locally" → exact commands to regenerate, Colab link for BERT ]
```

Add an info box that explains *why* the dashboard keeps the RF on handcrafted
features even if BERT wins: every score can be explained in terms of 6 human-readable
signals. That trade-off is the system's design choice, and the novelty
section (09) uses it.

## Commits (in order)

1. `feat(eval): add reusable compute_metrics with ROC/PR curve points`
2. `refactor(eval): evaluate_model delegates to compute_metrics`
3. `feat(models): add TF-IDF and hybrid feature builders`
4. `feat(models): add model zoo with RF, LR and SVM specs`
5. `feat(experiments): add model comparison script with per-model predictions`
6. `feat(models): add DistilBERT fine-tuning Colab notebook`
7. `feat(experiments): merge BERT predictions into model comparison`
8. `feat(charts): add model comparison chart builders`
9. `feat(dashboard): add Model Comparison page`
10. `test(models): add model zoo smoke tests`
11. `chore(results): add model comparison results on full dataset`

## Done when

- All 9 classical models plus DistilBERT appear in `results/model_comparison.json`, with metrics computed by the same function on the same 8,083 test rows.
- The page renders, and the filters work.
- `models/tfidf_lr.joblib` is produced (05 needs it), and git ignores it.

## Report notes

- Table: model × metric (exported by 07). Add one sentence each on hyperparameters and feature sets.
- Discuss the trade-off honestly. BERT and TF-IDF models will probably beat the 6-feature RF, and that's expected. The system's contribution is explainable credibility feeding summarization, not the top leaderboard score.
- Report the hybrid result. If hybrid > tfidf, the handcrafted credibility signals add information beyond the words.
