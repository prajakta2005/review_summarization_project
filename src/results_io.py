"""
results_io.py
-------------
Reads and writes experiment outputs in results/<name>.json.

Heavy experiments (BERT, cross-validation, ablation...) run offline as
scripts in experiments/ and the dashboard pages only READ their output.
Every file uses the same envelope so the Experimental Results page can
show where each number came from:

    {"meta": {name, generated_at, git_commit, dataset, n_rows, split, script},
     "data": {...experiment-specific...}}
"""

import json
import os
import subprocess
from datetime import datetime, timezone

from splits import SEED, TEST_SIZE

RESULTS_DIR = "results"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def result_path(name: str) -> str:
    return os.path.join(RESULTS_DIR, f"{name}.json")


def save_result(name: str, data: dict, dataset: str, n_rows: int, script: str) -> str:
    payload = {
        "meta": {
            "name": name,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "dataset": dataset,
            "n_rows": int(n_rows),
            "split": {"test_size": TEST_SIZE, "seed": SEED},
            "script": script,
        },
        "data": data,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = result_path(name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_result(name: str):
    """Returns the full {"meta", "data"} dict, or None if the experiment
    hasn't been run yet."""
    path = result_path(name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def list_results() -> list:
    """Metadata of every results/<name>.json, for the reproducibility panel."""
    if not os.path.isdir(RESULTS_DIR):
        return []
    metas = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "meta" in payload:
            metas.append(payload["meta"])
    return metas
