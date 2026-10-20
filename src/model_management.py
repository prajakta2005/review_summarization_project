"""
model_management.py
--------------------
Module 11: Model Management. Versioned training, saving, and loading.
"""

import json
import os
from datetime import datetime, timezone
import joblib

from credibility_model import train_model
from features import FEATURE_COLUMNS

MODELS_DIR = "models"
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")


def _load_registry() -> list:
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def _save_registry(registry: list):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def train_and_register(df, label_source: str = "unknown", notes: str = "") -> dict:
    clf, X_test, y_test = train_model(df)
    accuracy = clf.score(X_test, y_test)
    version_id = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"credibility_{version_id}.joblib")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, model_path)

    registry = _load_registry()
    entry = {
        "version_id": version_id, "model_path": model_path,
        "trained_at": datetime.now(timezone.utc).isoformat(), "training_rows": len(df),
        "label_source": label_source, "holdout_accuracy": round(accuracy, 4),
        "feature_columns": FEATURE_COLUMNS, "notes": notes,
    }
    registry.append(entry)
    _save_registry(registry)
    return entry


def list_versions() -> list:
    return _load_registry()


def load_model_by_version(version_id: str):
    registry = _load_registry()
    matches = [e for e in registry if e["version_id"] == version_id]
    if not matches:
        raise ValueError(f"No registered model with version_id={version_id}")
    return joblib.load(matches[0]["model_path"]), matches[0]


def load_latest_model():
    registry = _load_registry()
    if not registry:
        raise ValueError("No models registered yet -- call train_and_register() first.")
    latest = registry[-1]
    return joblib.load(latest["model_path"]), latest