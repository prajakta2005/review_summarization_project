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

from dashboard import require_pipeline
from summarizer import credibility_aware_summary
from comparison import compare_products
from evaluation import evaluate_model
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

pf, nf = st.columns(2)
with pf:
    st.markdown("**✅ Positive contributing factors**")
    for f in row["positive_factors"]:
        st.markdown(f"- {f}")
with nf:
    st.markdown("**⚠️ Negative contributing factors**")
    for f in row["negative_factors"]:
        st.markdown(f"- {f}")

st.markdown("**Natural-language explanation:**")
st.info(row["explanation_text"])

use_gemini = st.checkbox("Also generate a richer explanation with Gemini", value=False)
if use_gemini:
    from llm_explainer import generate_llm_explanation_safe
    with st.spinner("Asking Gemini to narrate the computed factors..."):
        llm_text = generate_llm_explanation_safe(
            row, row["positive_factors"], row["negative_factors"],
            fallback_text=row["explanation_text"]
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
with st.expander("📊 Model Performance Evaluation"):
    metrics = evaluate_model(clf, df)
    if metrics["label_source"] == "real_ground_truth":
        st.success("Evaluated on REAL ground-truth labels -- safe to cite in your report.")
    else:
        st.warning("Evaluated on heuristic pseudo-labels -- demo only, do NOT cite as a real benchmark.")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", metrics["accuracy"])
    m2.metric("Precision", metrics["precision"])
    m3.metric("Recall", metrics["recall"])
    m4.metric("F1 Score", metrics["f1_score"])
    m5.metric("ROC-AUC", metrics["roc_auc"] if metrics["roc_auc"] is not None else "N/A")
    st.text("Confusion matrix (rows=actual, cols=predicted, order=[not_credible, credible]):")
    st.write(pd.DataFrame(metrics["confusion_matrix"], columns=["Pred: Not Credible", "Pred: Credible"], index=["Actual: Not Credible", "Actual: Credible"]))
    st.text(metrics["classification_report"])

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
