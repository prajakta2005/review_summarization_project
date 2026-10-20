# 08 — System Output: live demo + screenshots

| | |
|---|---|
| **Owner** | Member B |
| **Phase** | 2 |
| **Branch** | `feature/system-output` |
| **Depends on** | 05 (SHAP/LIME for the demo). Take screenshots **last**, after 07 and 09 are merged. |
| **Dashboard page** | `pages/09_Live_Demo.py` |
| **Outputs** | `docs/screenshots/*.png`, `docs/SYSTEM_OUTPUT.md` |

## Goal

The report needs screenshots that show **credibility scores, explanations and
generated summaries**. Two parts:

1. **Live Demo page**: one screen where anyone (an examiner, a viva panel) can type a review and see the whole system work: score → reasons → where it would land in a summary. It's also the best source for screenshots.
2. **Automated screenshots**: a script that opens every dashboard page in a headless browser and saves full-page PNGs. Re-running after any UI change refreshes them all, and they're consistent (same window size and data).

## Files to add

| File | Purpose |
|---|---|
| `src/single_review.py` | `score_single_review(text, rating, category, clf, ref_stats) -> dict`. Builds a 1-row DataFrame, runs `preprocess` (without the dedup step), `extract_features`, `score_reviews`, the SHAP row explanation and the z-score explanation, and returns everything the page needs. `ref_stats` = feature stats of the loaded dataset, for the z-scores. |
| `pages/09_Live_Demo.py` | Dashboard (below) |
| `scripts/capture_screenshots.py` | Starts `streamlit run app.py --server.headless true --server.port 8599` as a subprocess, waits for it, then uses Playwright (Chromium, viewport 1440×900) to visit Home, picks *Kaggle full*, visits every page in `pages/`, waits until `[data-testid="stStatusWidget"]` is gone, and saves `docs/screenshots/<NN>_<page>.png` full-page. Also captures named states: Live Demo with a fake example and a genuine example, Explainability waterfall, Summarization side-by-side. Stops the server at the end. |
| `requirements-dev.txt` | add `playwright` (then run `python -m playwright install chromium` once) |
| `docs/SYSTEM_OUTPUT.md` | Screenshot gallery with figure numbers and one-line captions, ready to paste into the report |
| `tests/test_single_review.py` | An obviously hype-y review ("BEST!!! AMAZING!!! LOVE!!!", rating 5) scores lower than a specific one ("Used it for 3 weeks, battery lasts 2 days, strap broke after 10 days", rating 3). Output has all keys. |

## Files to change

| File | Change |
|---|---|
| `src/preprocessing.py` | `preprocess(df, dedupe=True)`: allow skipping dedup for single-row scoring |
| `pages/10_Review_Explorer.py` | Polish for screenshots: use `st.columns` cards for the score (big number + coloured label), and remove the debug-looking full-width table from the default view (move it into an expander) |
| `app.py` (Home) | Add a short "How to use this dashboard" block with `st.page_link`s to Live Demo, Summarization and Experimental Results |
| `.streamlit/config.toml` (new, tracked) | A theme (`primaryColor`, `font = "sans serif"`), so screenshots look consistent. Make sure `.gitignore` ignores only `secrets.toml`, not the folder. |

## Dashboard — `pages/09_Live_Demo.py`

```
Live Demo — score any review
┌ Review text ─────────────────────────────────────────┐  Rating ★★★★☆ (slider 1–5)
│ I've used this blender daily for 3 weeks. The motor  │  Category ▾ Home_and_Kitchen_5
│ is strong but the lid cracked on day 12...           │
└──────────────────────────────────────────────────────┘
[ Try: 🟢 genuine example ] [ 🔴 fake example ] [ 🎲 random from test set ]   [ Analyse ▶ ]

┌──────── Credibility ────────┐  ┌──────── Why? ─────────────────────────────┐
│  (Plotly gauge 0–100)  78%  │  │ SHAP waterfall (6 features)               │
│  HIGH · confidence 81%      │  │ ✅ specific details (numbers, timeframes) │
│  (true label if from test)  │  │ ✅ rating matches text tone               │
└─────────────────────────────┘  │ ⚠️ ...                                    │
                                 └───────────────────────────────────────────┘
Words that mattered (LIME text): highlighted review text, green/red
Plain-English explanation: st.info(...)    [☐ Ask Gemini for a richer explanation]

Feature values vs dataset:  table feature | this review | dataset mean | genuine mean | fake mean

Where would this review land?
  "In a credibility-aware summary of Home_and_Kitchen_5, this review would get weight 0.78
   (top 15% of the category); in a standard summary it gets the same weight as every other review."
```

The example buttons fill the text area from `st.session_state`. Pick the
"genuine" and "fake" examples from the **test split**, so the true label can be
shown next to the prediction. Use the `review_id`s for that.

## Screenshot list (`docs/SYSTEM_OUTPUT.md`)

| # | File | Shows |
|---|---|---|
| S1 | `01_home.png` | Overview KPIs on the full dataset |
| S2 | `02_live_demo_genuine.png` | High-credibility review, score + SHAP reasons |
| S3 | `03_live_demo_fake.png` | Low-credibility review, score + reasons + LIME words |
| S4 | `04_explainability_waterfall.png` | SHAP waterfall + LIME side by side |
| S5 | `05_summarization_side_by_side.png` | Standard vs credibility-aware summary, fake sentences marked |
| S6 | `06_review_explorer.png` | Product summary + low-credibility flags |
| S7 | `07_model_comparison.png` | Leaderboard |
| S8 | `08_performance.png` | Confusion matrix + metrics |
| S9 | `09_ablation.png` | Leave-one-out chart |
| S10 | `10_experimental_results.png` | Headline numbers |
| S11 | `11_dataset.png` | Dataset explorer |
| S12 | `12_novelty.png` | Novelty page with pipeline diagram |

S2, S3 and S5 are the three the brief asks for (credibility scores,
explanations, summaries). The rest support the other chapters.

## Commits (in order)

1. `feat(core): allow preprocessing without deduplication for single reviews`
2. `feat(core): add single-review scoring with SHAP and z-score explanations`
3. `feat(dashboard): add Live Demo page with gauge, SHAP and LIME output`
4. `style(dashboard): polish Review Explorer and add Home navigation for demos`
5. `chore(dashboard): add Streamlit theme config`
6. `feat(scripts): add Playwright screenshot capture for all dashboard pages`
7. `test(core): add single-review scoring tests`
8. `docs: add system output screenshots and gallery` (run the script after 07 and 09 are merged)

## Done when

- The Live Demo scores any typed review in under 2 seconds and shows score, SHAP reasons, LIME words and explanation text.
- `python scripts/capture_screenshots.py` produces all 12 PNGs with no manual clicking.
- `docs/SYSTEM_OUTPUT.md` shows every screenshot with a caption.

## Report notes

- "System Output" section: S2, S3 and S5, each with a 2–3 line caption explaining what the reader sees.
- Use the Live Demo for the viva. Rehearse one genuine and one fake example, and have the 🎲 button ready for "try a random one".
