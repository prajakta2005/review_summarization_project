# 00 — Foundation: multipage dashboard + shared plumbing

| | |
|---|---|
| **Owner** | Member A (Member B reviews) |
| **Branch** | `feature/foundation-multipage` |
| **Depends on** | nothing — merge this first |
| **Unblocks** | every other feature |

## Why this comes first

Today `app.py` is a single 220-line page that does everything. If two people
each add four features to it, every PR will conflict. Foundation:

1. Moves the pipeline into a reusable, cached function (`src/pipeline.py`).
2. Turns the app into a Streamlit multipage app, where each feature is its own file in `pages/`.
3. Adds the shared pieces every feature needs: one train/test split, one results format, and one place for charts.

Foundation adds **no new features**. When it's merged, the app must behave
exactly as it does today, just spread across pages.

## Files to add

| File | Purpose |
|---|---|
| `src/pipeline.py` | `load_raw()` + `run_pipeline()` — the existing preprocess → features → train → score → explain flow, moved out of `app.py` |
| `src/dashboard.py` | Streamlit helpers every page uses: `get_pipeline()`, `require_pipeline()`, `show_result_or_hint()` |
| `src/splits.py` | `train_test_indices()` — the single stratified split used by all models and experiments |
| `src/results_io.py` | `save_result()` / `load_result()` — reads/writes `results/<name>.json` with a metadata envelope |
| `src/charts/__init__.py` | Empty package; each later feature adds `src/charts/<feature>.py` |
| `pages/10_Review_Explorer.py` | The current product summary, comparison, drilldown, table, evaluation expander and storage expander, moved from `app.py` |
| `experiments/__init__.py` | Empty; makes `experiments/` a package for the scripts later features add |
| `results/.gitkeep` | Keeps the results folder in git |
| `models/.gitkeep` | Keeps `models/` in git (this was the cause of the `FileNotFoundError` we just fixed) |
| `tests/conftest.py` | Adds `src/` to `sys.path` and provides a small `sample_df` fixture |
| `tests/test_pipeline.py` | Smoke test: pipeline runs on the sample CSV and returns the expected columns |
| `requirements-dev.txt` | `pytest` |

## Files to change

| File | Change |
|---|---|
| `app.py` | Becomes **Home**: page config, title, data-source picker (sample / full Kaggle / upload + max-rows slider), runs `get_pipeline()`, shows the 4 overview metrics and 2 overview charts that are there today. Everything else moves to `pages/10_Review_Explorer.py`. |
| `src/credibility_model.py` | `train_model()` uses `splits.train_test_indices()` instead of its own `train_test_split(test_size=0.25)`. Already done in the working tree: `os.makedirs` before `joblib.dump`. |
| `src/evaluation.py` | `evaluate_model()` uses the same `train_test_indices()`, so it evaluates on exactly the rows `train_model()` held out |
| `.gitignore` | Change `models/*.joblib` to keep ignoring binaries and add `models/bert/`, `results/predictions/*.joblib`, `.streamlit/secrets.toml` |
| `README.md` | "Run it" section: mention the pages, `pytest`, and point to `docs/README.md` |

## Implementation steps

### 1. `src/splits.py`

```python
TEST_SIZE = 0.2
SEED = 42

def train_test_indices(y: pd.Series, test_size=TEST_SIZE, seed=SEED):
    """Stratified split on the label. Returns (train_idx, test_idx) as
    numpy arrays of positional indices. Falls back to an unstratified
    split when y has a single class (tiny sample dataset)."""

def export_split(df, path="results/split_indices.json"):
    """Writes {"train": [...review_id...], "test": [...]} so the BERT
    Colab notebook trains/tests on the identical rows."""
```

Use `review_id`, not the positional index, in `export_split`. `preprocess()`
drops duplicates and resets the index, but `review_id` stays stable.

### 2. `src/results_io.py` <a id="results-contract"></a>

Every experiment writes the same envelope:

```json
{
  "meta": {
    "name": "model_comparison",
    "generated_at": "2026-10-21T10:15:00Z",
    "git_commit": "abc1234",
    "dataset": "kaggle_full",
    "n_rows": 40412,
    "split": {"test_size": 0.2, "seed": 42},
    "script": "experiments/run_model_comparison.py"
  },
  "data": { "...feature-specific, documented in each feature doc..." }
}
```

```python
RESULTS_DIR = "results"
def save_result(name: str, data: dict, dataset: str, n_rows: int, script: str) -> str
def load_result(name: str) -> dict | None      # None if the file doesn't exist yet
```

Get `git_commit` with `subprocess.run(["git", "rev-parse", "--short", "HEAD"])`,
wrapped in try/except so it returns `"unknown"` outside a repo.

### 3. `src/pipeline.py`

```python
DATA_SOURCES = {
    "sample": "data/sample_reviews.csv",
    "kaggle_full": "data/fake reviews dataset.csv",
}

def load_raw(source: str, uploaded_df: pd.DataFrame | None = None,
             max_rows: int | None = None) -> pd.DataFrame:
    """Loads + schema-detects (moves the Kaggle auto-detect block from app.py
    lines 50-73 here) + optional stratified down-sampling."""

@dataclass
class PipelineResult:
    df: pd.DataFrame          # features
    clf: object               # trained RF
    scored: pd.DataFrame
    explained: pd.DataFrame
    train_idx: np.ndarray
    test_idx: np.ndarray

def run_pipeline(raw_df: pd.DataFrame) -> PipelineResult
```

This file must **not** import streamlit. Experiment scripts import it too.

### 4. `src/dashboard.py`

```python
@st.cache_resource(show_spinner="Running pipeline...")
def get_pipeline(source: str, max_rows: int | None, upload_key: str | None) -> PipelineResult

def require_pipeline() -> PipelineResult:
    """Called at the top of every page. Reads st.session_state['data_source'],
    ['max_rows'], ['uploaded_df'] (set on Home; default = sample) and returns
    the cached PipelineResult."""

def show_result_or_hint(name: str, script: str) -> dict | None:
    """load_result(name); if missing, st.info('Run `python {script}` to
    generate this section') and return None."""
```

In a multipage app, widgets placed on Home don't appear on other pages. That's
why the data choice lives in `st.session_state`, and `require_pipeline()` reads
it. Pages then show `Dataset: kaggle_full (40,412 rows)` as a caption at the top.

### 5. Split `app.py`

- `app.py` (Home): everything up to and including the "Dashboard Overview"
  section (current lines 32–109), rewritten to use `get_pipeline()`. Change the
  current `MAX_ROWS = 3000` hard cap to a sidebar slider, with
  `None` = all rows. The full set takes about 11 s cold and is instant after caching.
- `pages/10_Review_Explorer.py`: current lines 111–222, starting with
  `res = require_pipeline()`, then `explained = res.explained`,
  `clf = res.clf`, `df = res.df`.

### 6. Tests

`tests/test_pipeline.py`:
- `run_pipeline(load_raw("sample"))` returns `explained` with `credibility_score`, `explanation_text`.
- `train_test_indices` is disjoint, covers all rows, and is the same across two calls.

## Dashboard

Home page after this change:

```
┌ Sidebar ──────────────┐  🔍 Explainable Consumer Review Intelligence System
│ Data source           │  Dataset: kaggle_full · 40,412 reviews · RF trained on 32,329
│ ( ) Sample (15)       │  ┌────────┬────────┬────────┬────────┐
│ (•) Kaggle full (40k) │  │ Total  │ Avg    │ High   │ Low    │
│ ( ) Upload CSV        │  │ 40,412 │ 61.2   │ 14,880 │ 9,301  │
│ Max rows [── all ──]  │  └────────┴────────┴────────┴────────┘
│                       │  [Credibility pie]     [Global feature importance]
│ Pages ▸ Novelty ...   │  "Use the pages in the sidebar to explore each module."
└───────────────────────┘
```

## Commits (in order)

1. `fix(model): create models/ directory before saving the classifier`
   (the change already in the working tree, plus `models/.gitkeep`)
2. `feat(core): add shared stratified train/test split in splits.py`
3. `refactor(model): use shared split in train_model and evaluate_model`
4. `feat(core): add results_io with metadata envelope for experiment outputs`
5. `refactor(core): extract pipeline into src/pipeline.py`
6. `feat(dashboard): add cached pipeline helpers in src/dashboard.py`
7. `refactor(dashboard): convert app to multipage and move explorer to pages/10_Review_Explorer.py`
8. `feat(dashboard): replace 3000-row cap with max-rows slider on Home`
9. `test(core): add pipeline and split smoke tests`
10. `chore: add charts/experiments packages, results dir, dev requirements and gitignore rules`
11. `docs: document multipage layout and test command in README`

## Done when

- `streamlit run app.py` shows Home plus a Review Explorer page that behaves like the old single page.
- Switching the data source on Home changes the data on Review Explorer.
- Full Kaggle dataset loads without the 3000-row cap.
- `pytest` passes.
