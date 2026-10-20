# 01 — Full Dataset (≈40,000 reviews)

| | |
|---|---|
| **Owner** | Member A |
| **Phase** | 1 |
| **Branch** | `feature/full-dataset` |
| **Depends on** | 00 Foundation |
| **Dashboard page** | `pages/02_Dataset.py` |
| **Results file** | `results/dataset_profile.json` |

## Goal

Test the system on the complete Kaggle Fake Reviews Dataset
(`data/fake reviews dataset.csv`), not the 15-row sample or the 3,000-row
demo cap. Then **describe the dataset properly**, because the report needs a
dataset section and every later result depends on it.

Measured on this repo today:

| | |
|---|---|
| Raw rows | 40,432 |
| After `preprocess()` (exact-duplicate removal) | **40,412** (20 duplicates removed) |
| Labels | `OR` (original / genuine) 20,215 · `CG` (computer-generated / fake) 20,197 |
| Categories (`product_id`) | 10 Amazon categories, 3,588–4,730 reviews each |
| Pipeline time, full set | ~11 s (preprocess 1 s, features 3.4 s, train 2 s, score+explain 4.5 s) |
| Current RF on full set | accuracy 0.753 · precision 0.718 · recall 0.836 · F1 0.772 · ROC-AUC 0.824 |

Those RF numbers are the **baseline** that features 02 and 04 compare against.

## Files to add

| File | Purpose |
|---|---|
| `src/dataset_profile.py` | `profile_dataset(df) -> dict`: class balance, per-category counts and label mix, rating distribution by label, text length stats by label, the 6 feature distributions by label, duplicate count, missing-value count |
| `experiments/run_dataset_profile.py` | Loads the full set via `pipeline.load_raw("kaggle_full")`, runs preprocess + features, calls `profile_dataset`, `save_result("dataset_profile", ...)`, and also calls `splits.export_split(...)` → `results/split_indices.json` |
| `src/charts/dataset.py` | Chart builders: `class_balance_bar`, `category_stacked_bar`, `rating_by_label_hist`, `length_by_label_box`, `feature_violin(feature)` |
| `pages/02_Dataset.py` | Dataset dashboard (layout below) |
| `tests/test_dataset_profile.py` | Profile on the sample df has the expected keys; label counts add up to `len(df)` |

## Files to change

| File | Change |
|---|---|
| `src/kaggle_adapter.py` | Keep `category` as its own column (today it is only copied into `product_id`), and keep the original `label` string as `source_label` (`OR`/`CG`) for display. Default `sample_n` in `adapt_kaggle_fake_reviews` becomes `None` (full set). |
| `src/preprocessing.py` | `drop_duplicates()` returns the number removed, and `preprocess()` stores it in `df.attrs["duplicates_removed"]` so the profile can report it |
| `src/pipeline.py` | `DATA_SOURCES["kaggle_full"]` is the default the experiment scripts use |
| `README.md` | Replace "download + run adapter + upload" with "the full dataset is bundled at `data/fake reviews dataset.csv`; choose *Kaggle full* on Home". Cite the dataset source (Salminen et al., 2022 — verify the exact reference). |

## `results/dataset_profile.json` → `data`

```json
{
  "n_raw": 40432, "n_after_preprocess": 40412, "duplicates_removed": 20,
  "label_counts": {"genuine": 20215, "fake": 20197},
  "categories": [{"category": "Books_5", "genuine": 2185, "fake": 2185}, "..."],
  "rating_by_label": {"genuine": {"1": 0, "2": 0, "...": 0}, "fake": {"...": 0}},
  "length_by_label": {"genuine": {"mean": 0, "median": 0, "p95": 0}, "fake": {"...": 0}},
  "feature_stats_by_label": {"specificity": {"genuine": {"mean": 0, "std": 0}, "fake": {"...": 0}}, "...": {}},
  "feature_hist_by_label": {"length": {"bins": [], "genuine": [], "fake": []}, "...": {}}
}
```

Store histograms as **pre-binned counts**, not 40k raw values, so the JSON stays small.

## Dashboard — `pages/02_Dataset.py`

The page uses the *live* data from `require_pipeline()` (whatever is chosen on
Home). It also shows a "Full-dataset reference" tab from `dataset_profile.json`,
so the full-set numbers are visible even when the sample is loaded.

```
Dataset Explorer                               Source: kaggle_full · 40,412 rows
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ Reviews │ Genuine │ Fake    │ Categ.  │ Dupes   │   ← st.metric row
│ 40,412  │ 20,215  │ 20,197  │ 10      │ 20      │
└─────────┴─────────┴─────────┴─────────┴─────────┘
[ Tabs: Overview | Categories | Text & Ratings | Feature Distributions | Raw Data ]

Overview:            class-balance bar  +  donut of categories
Categories:          stacked bar genuine/fake per category (sorted)
Text & Ratings:      rating histogram by label  |  length box plot by label
Feature Distr.:      selectbox(feature) → overlaid violin genuine vs fake
                     + table of mean/std per label + "separation" (Cohen's d)
Raw Data:            filter by category / label / rating → st.dataframe, CSV download
```

The Cohen's d column on the Feature Distributions tab is a cheap preview of
which features separate the classes. Feature 04 (ablation) confirms it properly.

## Implementation notes

- `extract_features` uses row-wise `.apply`. 3.4 s on 40k is fine, so don't optimize yet.
- On the live page, sample 5,000 rows before drawing violins and histograms, so the browser stays fast. Show the counts from the full data.
- Map labels for display in one place: `{1: "genuine", 0: "fake"}`, i.e. `ground_truth_label == 1` ⇔ `OR`.

## Commits (in order)

1. `feat(data): keep category and source label columns in kaggle adapter`
2. `feat(data): report number of removed duplicates from preprocessing`
3. `feat(data): add dataset profiling module`
4. `feat(experiments): add dataset profile script and export shared split indices`
5. `feat(charts): add dataset chart builders`
6. `feat(dashboard): add Dataset explorer page`
7. `test(data): add dataset profile tests`
8. `chore(results): add full-dataset profile and split indices`
   (commit the generated `results/dataset_profile.json` + `results/split_indices.json`)
9. `docs: describe bundled full dataset in README`

## Done when

- `python experiments/run_dataset_profile.py` writes both results files from the full 40,412 rows.
- The Dataset page shows correct counts for *Sample* and for *Kaggle full*.
- The README no longer says the system is tested on 15 reviews.

## Report notes

- Dataset section: source, size, 50/50 balance, 10 categories, what `CG` vs `OR` means, preprocessing (URL removal, whitespace, 20 exact duplicates removed).
- State clearly that `product_id` = Amazon category in this dataset. The product-level summaries are really category-level summaries.
- Include the class-balance and per-category charts (exported in 07).
