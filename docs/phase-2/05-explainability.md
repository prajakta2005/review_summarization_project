# 05 — Explainability with SHAP and LIME

| | |
|---|---|
| **Owner** | Member B |
| **Phase** | 2 |
| **Branch** | `feature/shap-lime-explainability` |
| **Depends on** | 00 Foundation for the page. The `src/` modules can be written before 00 merges. Text-LIME is better with `models/tfidf_lr.joblib` from 02, but has a fallback. |
| **Dashboard page** | `pages/05_Explainability.py` |
| **Results file** | `results/explainability_eval.json` |

## Goal

Today, `explainability.py` explains a review by comparing its feature values
with the dataset mean (z-scores). That is transparent, but it **doesn't look at
the model**: two reviews with the same z-scores get the same explanation even if
the model treats them differently. Add two model-based explainers:

| Method | Explains | Model | Output |
|---|---|---|---|
| **SHAP** (`shap.TreeExplainer`) | the deployed RF | `rf_handcrafted` | Exact per-feature contribution to *this* review's credibility probability. Global importance = mean \|SHAP\|. |
| **LIME tabular** (`lime.lime_tabular`) | the deployed RF | `rf_handcrafted` | Local linear approximation, as a second opinion on SHAP |
| **LIME text** (`lime.lime_text`) | the TF-IDF + LR text model | `models/tfidf_lr.joblib` | **Which words** pushed a review toward genuine or fake, highlighted in the text |

Then **evaluate the explanations**, not just display them:

- **Agreement**: on 500 random test reviews, how often do z-score, SHAP and LIME rank the same feature as #1? Report top-3 overlap (Jaccard) and Kendall τ.
- **Faithfulness (deletion test)**: replace each review's top-k features by the dataset mean and measure the drop in predicted probability. A faithful method causes a bigger drop than removing k random features.
- **Class contrast**: mean SHAP per feature for genuine vs fake reviews. What makes the model call something fake?

Class convention: class `1` = genuine/credible. Positive SHAP or LIME weight
means "pushes toward credible", and negative means "pushes toward fake".

## Files to add

| File | Purpose |
|---|---|
| `src/shap_explainer.py` | `get_tree_explainer(clf)` (cached), `shap_values(clf, X) -> np.ndarray` (class-1 slice, shape n×6), `global_shap_importance(clf, X_sample) -> pd.Series`, `shap_row_factors(shap_row, x_row) -> list[(feature, value, contribution)]` |
| `src/lime_explainer.py` | `lime_tabular_explain(clf, X_train, x_row, num_features=6)`, `load_text_model()` (loads `models/tfidf_lr.joblib`, or trains a TF-IDF+LR on the current df in about 5 s if missing), `lime_text_explain(text_model, text, num_features=10) -> list[(word, weight)]`, `highlight_html(text, word_weights) -> str` |
| `src/explanation_eval.py` | `method_agreement(...)`, `deletion_faithfulness(...)`, `class_contrast(...)` |
| `experiments/run_explainability_eval.py` | Full dataset: trains/loads the RF on the shared split, computes global SHAP on the test split, runs the evaluation on 500 sampled test reviews → `results/explainability_eval.json` |
| `src/charts/explainability.py` | `shap_bar(global)`, `shap_beeswarm(shap_vals, X)` (Plotly strip, colour = feature value), `shap_waterfall(row_factors, base_value)`, `lime_bar(weights)`, `agreement_heatmap`, `faithfulness_curve`, `class_contrast_bar` |
| `pages/05_Explainability.py` | Dashboard (below) |
| `tests/test_explainability.py` | For a tree model, SHAP values + base value equal the predicted probability (tolerance 1e-6). LIME returns 6 features. `highlight_html` escapes HTML. |

## Files to change

| File | Change |
|---|---|
| `src/explainability.py` | Add `explain_review_shap(row, shap_row)`, which returns the **same dict** as `explain_review` (`positive_factors`, `negative_factors`, `explanation_text`), built from SHAP sign and magnitude and reusing `FEATURE_MEANING`. Add `method: str = "zscore"` and `clf=None` to `explain_dataframe()`. `"shap"` computes SHAP for all rows in one vectorized call. Keep `"zscore"` as the default, so nothing else breaks. |
| `src/pipeline.py` | `run_pipeline(raw_df, explain_method="zscore")` is passed through to `explain_dataframe` |
| `pages/10_Review_Explorer.py` | Drilldown gets a radio button, **Explanation method: z-score / SHAP**, and shows the SHAP waterfall for the selected review when SHAP is chosen |
| `app.py` (Home) | The global importance chart gets a toggle: RF impurity / mean \|SHAP\| |
| `src/llm_explainer.py` | When SHAP factors are available, pass them (with signs and magnitudes) to Gemini instead of the z-score phrases, so the narrated explanation is grounded in the model |
| `requirements.txt` | `shap`, `lime`, `matplotlib` |

Note: `shap` pulls in `numba`. If installing on Windows fails, use
`pip install shap --prefer-binary`.

## `results/explainability_eval.json` → `data`

```json
{
  "n_explained": 500,
  "global_shap": {"specificity": 0.0, "length": 0.0, "...": 0.0},
  "class_contrast": {"genuine": {"specificity": 0.0}, "fake": {"specificity": 0.0}},
  "agreement": {
    "top1_match": {"zscore_vs_shap": 0.0, "shap_vs_lime": 0.0, "zscore_vs_lime": 0.0},
    "top3_jaccard": {"zscore_vs_shap": 0.0, "shap_vs_lime": 0.0, "zscore_vs_lime": 0.0},
    "kendall_tau":  {"zscore_vs_shap": 0.0, "shap_vs_lime": 0.0, "zscore_vs_lime": 0.0}
  },
  "faithfulness": {
    "k": [1, 2, 3],
    "prob_drop": {"shap": [0.0, 0.0, 0.0], "lime": [], "zscore": [], "random": []}
  },
  "examples": [{"review_id": 123, "label": "fake", "score": 0.12, "shap": {}, "lime_words": [["amazing", -0.08]]}]
}
```

Feature 04 reads `global_shap` for its ranking comparison. Keep the key name.

## Dashboard — `pages/05_Explainability.py`

```
Explainability — SHAP & LIME
[ Tabs: Global | Single Review | Words (LIME text) | Method Agreement | Genuine vs Fake ]

Global:
  [ mean |SHAP| bar ]              [ SHAP beeswarm: x = SHAP value, y = feature,
                                     colour = feature value (low→high), 2,000 sampled rows ]
  caption: "Compare with ablation →" (page_link to 07)

Single Review:
  selectbox(review)  ·  [🎲 random fake] [🎲 random genuine] [🎲 misclassified]
  ┌ Score 23% LOW ┬ True label: fake ✔ ┐
  [ SHAP waterfall: base value → each feature's push → final probability ]
  [ LIME tabular bar ]   side by side  [ z-score factors (current method) ]
  natural-language explanation (SHAP-based)

Words (LIME text):
  same review, text with words highlighted green (→ genuine) / red (→ fake),
  rendered via st.markdown(highlight_html(...), unsafe_allow_html=True)
  + bar of top-10 words · free-text box "explain your own sentence"

Method Agreement:
  3×3 heatmap of top-1 agreement · faithfulness line chart (prob drop vs k, per method)

Genuine vs Fake:
  diverging bar: mean SHAP per feature for genuine vs fake reviews
```

SHAP on the RF (200 trees, depth 4, 6 features) takes seconds for thousands
of rows. Compute it live for the loaded data and cache it with `st.cache_data`.
LIME is slower (about 0.2–1 s per review), so run it **only** for the selected
review, on click.

## Commits (in order)

1. `feat(explain): add SHAP TreeExplainer wrapper for the credibility model`
2. `feat(explain): add LIME tabular and text explainers with HTML highlighting`
3. `feat(explain): support SHAP-based natural-language explanations in explain_dataframe`
4. `feat(explain): add explanation agreement and faithfulness evaluation`
5. `feat(experiments): add explainability evaluation script`
6. `feat(charts): add SHAP waterfall, beeswarm and LIME chart builders`
7. `feat(dashboard): add Explainability page`
8. `feat(dashboard): add SHAP option to Review Explorer drilldown and Home importance chart`
9. `feat(llm): ground Gemini explanations in SHAP contributions`
10. `test(explain): add SHAP additivity and LIME tests`
11. `chore(results): add explainability evaluation results`

## Done when

- Every review in the Review Explorer can be explained by z-score or SHAP.
- The Explainability page shows global SHAP, a per-review waterfall, LIME (tabular + word highlighting), and the agreement/faithfulness results.
- `results/explainability_eval.json` exists, and the 04 Rankings tab picks it up.

## Report notes

- Explain *why* there are three methods: z-score (model-agnostic, but not faithful to the model), SHAP (exact for trees, theoretically grounded), LIME (local surrogate, also works for the text model).
- Report the agreement and faithfulness numbers. SHAP should beat the random baseline in the deletion test. If z-score explanations disagree with SHAP often, that's a strong argument for the upgrade.
- Figures: one waterfall for a fake review, one LIME word highlight, global SHAP bar.
- Cite Lundberg & Lee (2017, SHAP) and Ribeiro et al. (2016, LIME).
