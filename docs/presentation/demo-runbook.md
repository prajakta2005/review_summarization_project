# Demo Runbook

## The day before

```bash
git checkout main && git pull            # or dev, whichever you present from
python -m pip install -r requirements.txt
python -m pytest -q                      # should say 52 passed
python -m streamlit run app.py
```

- Open http://localhost:8501, choose **Kaggle fake-reviews dataset (≈40k reviews)** on Home and
  wait for the first load (~12 s). Click through every page once so the caches are warm.
- Optional: put `GEMINI_API_KEY=...` in a `.env` file to show the Gemini explanation. Without it,
  the app falls back to the built-in explanation, which is fine.
- Have [`docs/screenshots/`](../screenshots/) open in another window as a backup.

## Ten minutes before

1. Close other heavy programs (the full dataset uses ~1.5 GB of RAM).
2. Start the app, pick the Kaggle dataset on **Home**, then open **Live Demo**, **Summarization**,
   **Explainability** and **Novelty** once each, so nothing computes during the talk.
3. Browser zoom 110–125% so the room can read it; sidebar open.

## The 6-minute demo script

| # | Page | Do | Say |
|---|---|---|---|
| 1 | **Home** | Point at the four numbers and the pie | "40,412 real Amazon reviews, half genuine and half computer-generated. Every page uses this data." |
| 2 | **Live Demo** | Click **🔴 Fake example** | "A review the model never saw. Credibility 12%. The waterfall shows why: repetitive wording pulled it down by 40 points." |
| 3 | Live Demo | Scroll to *Words that mattered* | "LIME highlights the words the text model reacted to." |
| 4 | Live Demo | Click **🟢 Genuine example** | "A genuine one scores high: specific details, varied wording. And in a summary its sentences get high weight." |
| 5 | Live Demo | Type your own hype review, e.g. *"BEST product EVER!!! love love love it, buy buy buy!!!"*, rating 5 | "You can type anything: the score and the reasons update instantly." |
| 6 | **Summarization** | Category *Books_5*, 5 sentences | "Same summarizer twice. Left: every review trusted equally, and all 5 sentences come from fake reviews. Right: weighted by credibility, and most of them now come from genuine reviews." |
| 7 | Summarization | *All categories* tab | "Across all 10 categories: 92% fake content down to 42%, ROUGE-L up from 0.31 to 0.41." |
| 8 | **Novelty** | *Trace a summary sentence*: pick a sentence | "Every summary sentence traces back to its review, its score and the reasons for that score." |
| 9 | **Experimental Results** | Headline numbers, then *Tables* | "Every number in the talk comes from here, and one command regenerates it all." |

If you have extra time: **Explainability → Method Agreement** (SHAP is the most faithful), and
**Ablation Study** (repetition and length carry the model).

## If something breaks

| Problem | Fix |
|---|---|
| Page shows a spinner for a long time | It's computing for the first time: talk over it, or switch to the screenshot |
| "No results yet" box on a page | `results/` is missing. Run `python experiments/run_all.py` (17 min), or show the screenshot |
| Gemini checkbox shows "unavailable" | No API key: say "it falls back to the built-in explanation" and move on |
| App won't start | Present from [`docs/screenshots/`](../screenshots/): S2, S3 and S5 cover the demo |
| Port in use | `python -m streamlit run app.py --server.port 8502` |
| `ModuleNotFoundError` on a page | The `streamlit` command uses another Python: run `python -m pip install -r requirements.txt` and `python -m streamlit run app.py` |

## Questions you'll probably get

- **"Why not just use BERT?"** BERT and TF-IDF models score higher (F1 ≈ 0.95), but they can't say
  *why* a review is fake in terms a shopper understands. The system needs explanations it can show
  and summaries it can trace. See [02-model-comparison.md](02-model-comparison.md).
- **"Isn't 42% contamination still high?"** Yes, and we say so. It's down from 92%, the deployed
  model is deliberately simple, and a hard filter gets it to 20% at a quality cost. A stronger
  credibility model plugs straight in. See [05-summarization.md](05-summarization.md).
- **"Did the model see the reviews it scored?"** Not in the reported results: summary scores use
  5-fold out-of-fold predictions, and every classifier is tested on the same held-out 20%.
- **"What does 'fake' mean here?"** Computer-generated reviews (label `CG`) from the Salminen et al.
  (2022) dataset. Human-written deceptive reviews may behave differently: listed as a limitation.
