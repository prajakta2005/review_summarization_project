# Feature Roadmap — Explainable Review Intelligence System

This folder is the implementation plan for the next round of features. Each
feature has its own file that lists the files to add, the files to change, the
branch to create, the commits to make (in order), and the dashboard page it adds.

## Team split

| | Member A — **Phase 1: Data & Models** | Member B — **Phase 2: Explainability, Summaries & Presentation** |
|---|---|---|
| Focus | Get honest numbers on the full dataset | Explain them, show the summarization gain, package the results |
| Features | [00 Foundation](00-foundation.md) (shared, do first)<br>[01 Dataset](phase-1/01-dataset.md)<br>[02 Model Comparison](phase-1/02-model-comparison.md)<br>[03 Performance Results](phase-1/03-performance-results.md)<br>[04 Ablation Study](phase-1/04-ablation-study.md) | [05 Explainability (SHAP + LIME)](phase-2/05-explainability.md)<br>[06 Summarization Comparison](phase-2/06-summarization.md)<br>[07 Experimental Results](phase-2/07-experimental-results.md)<br>[08 System Output](phase-2/08-system-output.md)<br>[09 Novelty](phase-2/09-novelty.md) |
| Produces | `results/dataset_profile.json`, `model_comparison.json`, `cv_results.json`, `ablation.json`, `results/predictions/*.csv` | `results/explainability_eval.json`, `summary_comparison.json`, `docs/figures/`, `docs/screenshots/`, novelty statement |

The phases are named by **ownership**, not strict order. After Foundation is
merged, both members work **in parallel**. Only features 07, 08 and 09 must wait
for Phase 1 results, because they aggregate them.

## Order of work and merge order

```
                    ┌─ A: 01 dataset ─► 02 model-comparison ─► 03 performance ─► 04 ablation ─┐
00 foundation (A) ──┤                                                                         ├─► B: 07 experimental-results ─► 08 system-output ─► 09 novelty
                    └─ B: 05 explainability ─► 06 summarization ───────────────────────────────┘
```

| # | Branch | Owner | Can start when | Rough effort |
|---|---|---|---|---|
| 00 | `feature/foundation-multipage` | A | now | 1 day |
| 01 | `feature/full-dataset` | A | 00 merged | 1 day |
| 02 | `feature/model-comparison` | A | 01 merged | 3–4 days (BERT on Colab) |
| 03 | `feature/performance-results` | A | 02 merged | 2 days |
| 04 | `feature/ablation-study` | A | 01 merged (best after 02) | 2 days |
| 05 | `feature/shap-lime-explainability` | B | src work now, page after 00 | 3 days |
| 06 | `feature/summary-comparison` | B | 00 merged | 3 days |
| 07 | `feature/experimental-results` | B | 02, 03, 04, 05, 06 merged | 2 days |
| 08 | `feature/system-output` | B | 05 merged; screenshots last | 2 days |
| 09 | `docs/novelty` | B | 06 merged (numbers from 07) | 1–2 days |

While 00 is in review, Member B can already write `src/shap_explainer.py` and
`src/lime_explainer.py` (pure library code, no page yet). Those files don't
touch anything 00 changes.

## Dashboard map (Streamlit multipage)

Foundation turns `app.py` into a multipage app. Streamlit finds every file in
`pages/` automatically. Each feature adds **its own page file**, so the two
members never edit the same page and don't hit merge conflicts on the UI.

| File | Sidebar label | Feature | Owner |
|---|---|---|---|
| `app.py` | Home | Data source picker + overview KPIs | A (00) |
| `pages/01_Novelty.py` | Novelty | 09 | B |
| `pages/02_Dataset.py` | Dataset | 01 | A |
| `pages/03_Model_Comparison.py` | Model Comparison | 02 | A |
| `pages/04_Performance.py` | Performance | 03 | A |
| `pages/05_Explainability.py` | Explainability | 05 | B |
| `pages/06_Summarization.py` | Summarization | 06 | B |
| `pages/07_Ablation_Study.py` | Ablation Study | 04 | A |
| `pages/08_Experimental_Results.py` | Experimental Results | 07 | B |
| `pages/09_Live_Demo.py` | Live Demo | 08 | B |
| `pages/10_Review_Explorer.py` | Review Explorer | existing dashboard, moved in 00 | A |

## Shared conventions (agree on these once)

**Two kinds of computation.** The **live pipeline** (preprocess → features →
RF → explanations) takes about 11 s on all 40,412 reviews, so pages run it
live and cache the result. **Heavy experiments** (BERT, 5-fold CV, ablation,
summary evaluation) run offline as scripts in `experiments/`. They write small
JSON/CSV files to `results/`, and the pages only *read* those files. That keeps
the dashboard fast, and anyone can reproduce the numbers by re-running the
scripts.

**Same split everywhere.** Every model and experiment gets its train/test rows
from `src/splits.py` (added in 00): stratified, `test_size=0.2`, `seed=42`. BERT
on Colab uses the same indices, exported to `results/split_indices.json`.
Without this, the comparison tables aren't valid.

**Results files** all use the envelope defined in
[00-foundation.md](00-foundation.md#results-contract):
`{"meta": {...}, "data": {...}}`. They're small, so they're committed to git.
Model binaries (`*.joblib`, BERT weights) are **not** committed.

**Charts.** Each feature puts its Plotly chart builders in its own module under
`src/charts/` (e.g. `src/charts/ablation.py`). Pages call them, and feature 07
calls the same functions to export report figures. That way there's one chart
definition and no duplicated plotting code.

**Dependencies.** Each feature adds only its own lines to `requirements.txt`.
If both branches change it, the merge conflict is a one-line fix. BERT
dependencies go in `requirements-bert.txt`, so the main app installs without
torch.

## Git workflow

```bash
git checkout main && git pull
git checkout -b feature/<name>          # branch name from the table above
# ... make the commits listed in the feature doc, in order ...
git push -u origin feature/<name>
# open a PR into main; the OTHER member reviews and merges
```

- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat(scope): ...`, `fix(...)`, `refactor(...)`, `docs(...)`, `test(...)`, `chore(...)`).
  Every feature doc gives the exact messages.
- Merge PRs with a **merge commit** (not squash), so these commit messages stay in the history.
- `main` must always run: `streamlit run app.py` works and `pytest` passes before you merge.
- Before opening a PR, rebase on `main` (`git pull --rebase origin main`) so conflicts are fixed on your branch, not on `main`.

## Definition of done (every feature)

1. Its dashboard page renders with **both** the sample dataset and the full Kaggle dataset.
2. If it has an experiment script, the script runs from the repo root and writes its `results/*.json`.
3. Its `tests/test_<feature>.py` passes.
4. Its "Report notes" section has been turned into text in the team's report draft.
