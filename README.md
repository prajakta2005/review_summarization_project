# Explainable Consumer Review Intelligence System

Working prototype demonstrating the patented workflow:
**Reviews → Preprocessing → Feature Extraction → Credibility Assessment →
Explainability → Credibility-Aware Summarization → Dashboard**

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501. It's a multipage app: **Home** (`app.py`) picks the
data source and shows the overview, and each module/feature has its own page in
`pages/` (listed in the sidebar). The data source chosen on Home is used by every page.

A bundled sample dataset (`data/sample_reviews.csv`) is loaded by default; you can
also upload your own CSV with columns:
`review_id, product_id, reviewer_id, rating, timestamp, review_text`.

Run the tests with:

```bash
pip install -r requirements-dev.txt
pytest
```

The feature roadmap (who builds what, branches, commits) is in [docs/README.md](docs/README.md).

## The full dataset (≈40k labeled reviews)

The bundled sample has only 15 hand-written rows. The full
**[Fake Reviews Dataset (mexwell/fake-reviews-dataset)](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)**
is bundled at `data/fake reviews dataset.csv`: 40,432 reviews (40,412 after removing
20 exact duplicates) from 10 Amazon categories, balanced 50:50 between `OR`
(original, human-written = genuine) and `CG` (computer-generated = fake). It was
introduced by Salminen et al. (2022), *Journal of Retailing and Consumer Services*
(verify the exact reference before citing).

To use it, choose **Kaggle fake-reviews dataset** on the Home page. All rows are used
by default (about 11 s on first load, then cached); the *Max rows* slider takes a
stratified sample if you want something faster. The **Dataset** page profiles it.

`credibility_model.py` automatically detects the `ground_truth_label` column the
adapter produces and trains on **real labels** instead of pseudo-labels, which is
what makes the accuracy/precision/recall/ROC-AUC numbers citable. Note that in this
dataset `product_id` is the Amazon category, so "product" summaries are category summaries.

To regenerate the full-dataset profile and the shared train/test split
(`results/split_indices.json`, also used by the BERT notebook):

```bash
python experiments/run_dataset_profile.py
```

## What each file does

| File | Module (per spec) |
|---|---|
| `src/preprocessing.py` | Review Preprocessing |
| `src/features.py` | Feature Extraction & Analysis |
| `src/credibility_model.py` | Credibility Assessment Engine |
| `src/explainability.py` | Explainable AI Module |
| `src/summarizer.py` | Automated + Credibility-Aware Summarization |
| `src/pipeline.py` | End-to-end pipeline used by the dashboard and experiment scripts |
| `src/splits.py` | The single shared train/test split (stratified, 20%, seed 42) |
| `src/results_io.py` | Reads/writes experiment outputs in `results/` |
| `src/dashboard.py` | Streamlit helpers shared by all pages |
| `app.py` | Dashboard Home (data source + overview) |
| `pages/` | One dashboard page per module/feature |
| `experiments/` | Offline experiment scripts that write `results/*.json` |

## Honest scope notes — read before your viva/demo

This was built in one session inside a sandboxed environment with **no internet
access to model hubs** (HuggingFace, NLTK downloads, etc.) and **no real labeled
fake-review dataset**. Two deliberate simplifications follow directly from that,
and you should be ready to explain them if asked:

1. **Sentiment analysis** uses a small hand-built positive/negative word lexicon,
   not a pretrained model. Swap in NLTK's VADER or a HuggingFace sentiment
   pipeline (`from transformers import pipeline`) once you have internet access —
   the rest of `features.py` doesn't need to change, only `sentiment_score()`.

2. **Credibility labels for training** are heuristic pseudo-labels (weak
   supervision) generated from the same features the model learns on — see the
   big comment at the top of `credibility_model.py`. This is a legitimate way to
   bootstrap a system before real labeled data exists, but any accuracy/precision
   numbers from this data are **not a real benchmark**. Before writing your
   evaluation section, retrain on a real labeled dataset (e.g. the Yelp/Amazon
   academic fake-review datasets, or your own team's hand-labeled sample).

3. **Summarization is extractive** (TF-IDF sentence ranking), not abstractive/LLM.
   This is actually defensible as a design choice for a credibility system —
   every summary sentence is a real, traceable sentence from a real review, so
   nothing is hallucinated. If you want abstractive summaries, swap
   `rank_sentences()` in `summarizer.py` for a call to an LLM API or a
   HuggingFace summarization pipeline.

## Not yet built (documented, not faked)

Per the original spec, these need real infrastructure/time investment beyond a
single session and are intentionally left as next steps rather than stubbed out
with fake code:
- Persistent database storage (currently everything is in-memory/CSV)
- External review-platform API integrations
- Model versioning / retraining pipeline
- Formal evaluation report (accuracy/precision/recall/ROC-AUC) — needs real
  labeled data first, see point 2 above

## Extending the feature set

`features.py` currently has 6 interpretable features (sentiment, length,
exclamation density, repetition ratio, specificity, rating-text consistency).
To add "semantic features using embeddings" as in the original spec, add a new
function there, add its name to `FEATURE_COLUMNS`, and add its interpretation
to `FEATURE_MEANING` / `HIGH_IS_CREDIBLE` in `explainability.py` — the model
and dashboard will pick it up automatically.
