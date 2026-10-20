"""
Review Explorer: product-level credibility-aware summaries, attribute
comparison, per-review drilldown, storage and model versioning.
(Moved here unchanged from the original single-page app.py.)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd

from dashboard import require_pipeline, get_shap_values
from summarizer import credibility_aware_summary
from comparison import compare_products
from storage import save_reviews, save_product_summary, DB_PATH
from model_management import train_and_register, list_versions

st.set_page_config(page_title="Review Explorer", layout="wide")
st.title("Review Explorer")

res = require_pipeline()
explained, clf, df = res.explained, res.clf, res.df

# --- Product-level credibility-aware summary ---
st.header("Credibility-Aware Summary by Product")
product_ids = explained["product_id"].unique()
selected_product = st.selectbox("Select product", product_ids)

summary = credibility_aware_summary(explained, product_id=selected_product)
st.subheader("Overall Summary")
st.write(summary["overall_summary"])
st.subheader("High-Credibility Consensus")
st.write(summary["high_credibility_consensus"])
if summary["low_credibility_flags"]:
    st.subheader("⚠️ Low-Credibility Reviews (surfaced, not deleted)")
    for flag in summary["low_credibility_flags"]:
        st.warning(f"Review #{flag['review_id']} ({flag['credibility_score']}%): {flag['excerpt']}")

# --- Review Comparison & Aggregation (module 9) ---
st.header("Review Comparison & Aggregation")
comparison = compare_products(explained, selected_product)
cc1, cc2 = st.columns(2)
with cc1:
    st.subheader("✅ Consensus attributes")
    if comparison["consensus"]:
        for item in comparison["consensus"]:
            st.markdown(f"- **{item['attribute']}**: mostly {item['sentiment']} ({item['mentions']} mention(s))")
    else:
        st.caption("No consensus attributes detected for this product.")
with cc2:
    st.subheader("⚡ Conflicting attributes")
    if comparison["conflicting"]:
        for item in comparison["conflicting"]:
            st.markdown(f"- **{item['attribute']}**: {item['positive']} positive vs {item['negative']} negative mentions")
    else:
        st.caption("No conflicting opinions detected for this product.")

# --- Review-level drilldown: Review -> Score -> Reasons -> Summary ---
st.header("Review-Level Drilldown")
review_ids = explained["review_id"].tolist()
selected_id = st.selectbox("Select a review", review_ids)
row = explained[explained["review_id"] == selected_id].iloc[0]

st.markdown(f"**Review text:** {row['review_text']}")
score_color = {"High": "green", "Medium": "orange", "Low": "red"}[row["credibility_label"]]
st.markdown(f"**Credibility: :{score_color}[{row['credibility_score']}% — {row['credibility_label']}]**  |  Confidence: {row['confidence']}%")

method = st.radio("Explanation method", ["SHAP (follows the model)", "z-score (original)"], horizontal=True,
                  help="SHAP uses the model's exact feature contributions; z-score compares the review "
                       "to the dataset average. See the Explainability page for how they compare.")
shap_factors = None
if method.startswith("SHAP"):
    from explainability import explain_review_shap
    from shap_explainer import base_value, shap_row_factors
    from features import FEATURE_COLUMNS
    from charts.explainability import shap_waterfall
    i = explained.index.get_loc(row.name)
    shap_row = get_shap_values(res)[i]
    explanation = explain_review_shap(row, shap_row, explained[FEATURE_COLUMNS].mean())
    shap_factors = shap_row_factors(shap_row, row)
else:
    explanation = {k: row[k] for k in ["positive_factors", "negative_factors", "explanation_text"]}

pf, nf = st.columns(2)
with pf:
    st.markdown("**✅ Positive contributing factors**")
    for f in explanation["positive_factors"]:
        st.markdown(f"- {f}")
with nf:
    st.markdown("**⚠️ Negative contributing factors**")
    for f in explanation["negative_factors"]:
        st.markdown(f"- {f}")

st.markdown("**Natural-language explanation:**")
st.info(explanation["explanation_text"])
if shap_factors:
    st.plotly_chart(shap_waterfall(shap_factors, base_value(clf)), use_container_width=True)

use_gemini = st.checkbox("Also generate a richer explanation with Gemini", value=False)
if use_gemini:
    from llm_explainer import generate_llm_explanation_safe
    with st.spinner("Asking Gemini to narrate the computed factors..."):
        llm_text = generate_llm_explanation_safe(
            row, explanation["positive_factors"], explanation["negative_factors"],
            fallback_text=explanation["explanation_text"]
        )
    st.markdown("**Gemini explanation:**")
    st.success(llm_text)

# --- Full table ---
st.header("All Reviews")
st.dataframe(
    explained[["review_id", "product_id", "rating", "credibility_score", "credibility_label", "confidence"]],
    use_container_width=True
)

# --- Performance Evaluation (module 12) ---
st.page_link("pages/04_Performance.py", label="📊 See full performance results (all models, confusion matrix, CV) →")

# --- Data Storage & Model Management (modules 10 & 11) ---
with st.expander("🗄️ Data Storage & Model Versioning"):
    st.caption(f"SQLite database: {DB_PATH}")
    if st.button("Save current results to database"):
        save_reviews(explained)
        for pid in explained["product_id"].unique():
            save_product_summary(pid, credibility_aware_summary(explained, product_id=pid))
        st.success(f"Saved {len(explained)} reviews and their summaries to {DB_PATH}.")

    st.caption("Model versions (each trained + registered run gets a timestamped version):")
    if st.button("Train & register a new model version"):
        label_src = "real_ground_truth" if "ground_truth_label" in df.columns and df["ground_truth_label"].notna().all() else "heuristic_pseudo_labels"
        entry = train_and_register(df, label_source=label_src, notes="registered from dashboard")
        st.success(f"Registered version {entry['version_id']} (accuracy={entry['holdout_accuracy']})")

    versions = list_versions()
    if versions:
        st.dataframe(pd.DataFrame(versions)[["version_id", "trained_at", "training_rows", "label_source", "holdout_accuracy"]], use_container_width=True)
    else:
        st.caption("No model versions registered yet.")
