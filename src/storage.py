"""
storage.py
----------
Module 10: Data Storage. SQLite persistence for reviews, features,
predictions, explanations, and summaries.
"""

import json
import sqlite3
from datetime import datetime, timezone
import pandas as pd

DB_PATH = "review_intel.db"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def init_db(db_path: str = DB_PATH):
    conn = get_connection(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id INTEGER PRIMARY KEY, product_id TEXT, reviewer_id TEXT,
            rating REAL, review_text TEXT, clean_text TEXT, sentiment REAL,
            length INTEGER, exclaim_density REAL, repetition_ratio REAL,
            specificity INTEGER, rating_consistency REAL, credibility_score REAL,
            credibility_label TEXT, confidence REAL, positive_factors TEXT,
            negative_factors TEXT, explanation_text TEXT, ingested_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS product_summaries (
            product_id TEXT PRIMARY KEY, overall_summary TEXT,
            high_credibility_consensus TEXT, low_credibility_flags TEXT, generated_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_reviews(explained_df: pd.DataFrame, db_path: str = DB_PATH):
    init_db(db_path)
    conn = get_connection(db_path)
    df = explained_df.copy()
    df["positive_factors"] = df["positive_factors"].apply(json.dumps)
    df["negative_factors"] = df["negative_factors"].apply(json.dumps)
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()
    cols = ["review_id", "product_id", "reviewer_id", "rating", "review_text", "clean_text",
            "sentiment", "length", "exclaim_density", "repetition_ratio", "specificity",
            "rating_consistency", "credibility_score", "credibility_label", "confidence",
            "positive_factors", "negative_factors", "explanation_text", "ingested_at"]
    df[cols].to_sql("reviews", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()


def save_product_summary(product_id: str, summary: dict, db_path: str = DB_PATH):
    init_db(db_path)
    conn = get_connection(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO product_summaries VALUES (?, ?, ?, ?, ?)",
        (product_id, summary["overall_summary"], summary["high_credibility_consensus"],
         json.dumps(summary["low_credibility_flags"]), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def load_reviews_from_db(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = get_connection(db_path)
    df = pd.read_sql("SELECT * FROM reviews", conn)
    conn.close()
    df["positive_factors"] = df["positive_factors"].apply(json.loads)
    df["negative_factors"] = df["negative_factors"].apply(json.loads)
    return df