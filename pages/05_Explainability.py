"""
Explainability: SHAP and LIME explanations for the credibility model,
plus an evaluation of how much the explanation methods agree and how
faithful they are to the model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import require_pipeline, get_shap_values, get_text_model, show_result_or_hint
from features import FEATURE_COLUMNS
from shap_explainer import base_value, shap_row_factors
from lime_explainer import lime_tabular_explain, lime_text_explain, highlight_html
from explainability import explain_review, explain_review_shap
from charts import explainability as charts

st.set_page_config(page_title="Explainability", layout="wide")
st.title("Explainability — SHAP & LIME")

res = require_pipeline()
df = res.explained
shap_vals = get_shap_values(res)
base = base_value(res.clf)
has_labels = "ground_truth_label" in df.columns and df["ground_truth_label"].notna().all()
pos_of = {rid: i for i, rid in enumerate(df["review_id"])}

tabs = st.tabs(["Global", "Single Review", "Words (LIME text)", "Method Agreement", "Genuine vs Fake"])

# --- Global ---
with tabs[0]:
    global_shap = dict(zip(FEATURE_COLUMNS, np.abs(shap_vals).mean(axis=0)))
    c1, c2 = st.columns([2, 3])
    c1.plotly_chart(charts.shap_bar(global_shap), use_container_width=True)
    c2.plotly_chart(charts.shap_beeswarm(shap_vals, df), use_container_width=True)
    st.caption(f"SHAP values for the deployed Random Forest on {len(df):,} reviews (beeswarm shows up to 2,000). "
               "Each value is how many probability points that feature moved one review's credibility.")
    st.page_link("pages/07_Ablation_Study.py", label="Compare with the ablation study →")

# --- pick a review (shared by Single Review and Words tabs) ---
if "xai_review_id" not in st.session_state or st.session_state["xai_review_id"] not in pos_of:
    st.session_state["xai_review_id"] = int(df["review_id"].iloc[0])

def pick(mask):
    ids = df.loc[mask, "review_id"]
    if len(ids):
        st.session_state["xai_review_id"] = int(ids.sample(1).iloc[0])

predicted = (df["credibility_score"] >= 50).astype(int)
truth = df["ground_truth_label"] if has_labels else predicted

with tabs[1]:
    b = st.columns([1, 1, 1, 2])
    if b[0].button("🎲 Random fake" if has_labels else "🎲 Random low score"):
        pick(truth == 0)
    if b[1].button("🎲 Random genuine" if has_labels else "🎲 Random high score"):
        pick(truth == 1)
    if has_labels and b[2].button("🎲 Misclassified"):
        pick(truth != predicted)
    chosen = b[3].number_input("Review ID", min_value=int(df["review_id"].min()),
                               max_value=int(df["review_id"].max()), step=1,
                               value=st.session_state["xai_review_id"])
    if chosen in pos_of:
        st.session_state["xai_review_id"] = int(chosen)
    else:
        st.warning(f"Review {chosen} isn't in the loaded data (it may have been a duplicate).")

    i = pos_of[st.session_state["xai_review_id"]]
    row = df.iloc[i]
    m = st.columns(3)
    m[0].metric("Credibility score", f"{row['credibility_score']:.1f}%", help="P(genuine) from the Random Forest")
    m[1].metric("Label", row["credibility_label"])
    if has_labels:
        true = "genuine" if row["ground_truth_label"] == 1 else "fake"
        correct = (row["credibility_score"] >= 50) == (row["ground_truth_label"] == 1)
        m[2].metric("True label", true, "model correct" if correct else "model wrong",
                    delta_color="normal" if correct else "inverse")
    st.markdown(f"> {row['review_text'][:600]}")

    factors = shap_row_factors(shap_vals[i], row)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.shap_waterfall(factors, base), use_container_width=True)
    with c2:
        with st.spinner("Running LIME..."):
            lime = lime_tabular_explain(res.clf, res.df.iloc[res.train_idx].head(5000), row)
        st.plotly_chart(charts.lime_bar([(d["condition"], d["weight"]) for d in lime],
                                        "LIME (tabular): local linear weights"), use_container_width=True)

    means = df[FEATURE_COLUMNS].mean()
    shap_text = explain_review_shap(row, shap_vals[i], means)["explanation_text"]
    z_text = explain_review(row, df[FEATURE_COLUMNS].describe())["explanation_text"]
    e1, e2 = st.columns(2)
    e1.markdown("**Explanation from SHAP (follows the model)**")
    e1.info(shap_text)
    e2.markdown("**Explanation from z-scores (original method)**")
    e2.info(z_text)

# --- Words (LIME text) ---
with tabs[2]:
    text_model = get_text_model(res)
    st.markdown("Which **words** pushed the TF-IDF text model toward genuine (green) or fake (red)? "
                "This explains the text model from the model comparison, not the Random Forest.")
    row = df.iloc[pos_of[st.session_state["xai_review_id"]]]
    with st.spinner("Running LIME on the text..."):
        words = lime_text_explain(text_model, row["clean_text"])
    p = text_model.predict_proba([row["clean_text"]])[0, 1]
    st.caption(f"Review #{row['review_id']} · text model P(genuine) = {p:.2f}")
    st.markdown(highlight_html(row["clean_text"], words), unsafe_allow_html=True)
    st.plotly_chart(charts.lime_bar(words, "Top words (LIME text)"), use_container_width=True)

    st.markdown("**Try your own text**")
    own = st.text_area("Review text", "Absolutely amazing!!! Best product ever, I love it so much, buy it now!")
    if own.strip():
        own_words = lime_text_explain(text_model, own)
        st.caption(f"Text model P(genuine) = {text_model.predict_proba([own])[0, 1]:.2f}")
        st.markdown(highlight_html(own, own_words), unsafe_allow_html=True)

# --- Method agreement (offline evaluation) ---
with tabs[3]:
    ev = show_result_or_hint("explainability_eval", "experiments/run_explainability_eval.py")
    if ev:
        d = ev["data"]
        st.caption(f"{d['n_explained']} random test reviews from {ev['meta']['dataset']} "
                   f"({ev['meta']['n_rows']:,} rows) · generated {ev['meta']['generated_at']}")
        metric = st.radio("Agreement measure", ["top1_match", "top3_jaccard", "kendall_tau"], horizontal=True,
                          format_func={"top1_match": "same #1 feature", "top3_jaccard": "top-3 overlap",
                                       "kendall_tau": "Kendall τ"}.get)
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.agreement_heatmap(d["agreement"], metric), use_container_width=True)
        c2.plotly_chart(charts.faithfulness_curve(d["faithfulness"]), use_container_width=True)
        pc = d["faithfulness"]["prob_change"]
        best = max((m for m in pc if m != "random"), key=lambda m: pc[m][0])
        st.markdown(
            f"- Resetting the single feature each method calls most important moves the prediction by "
            f"**{pc['shap'][0]:.3f}** (SHAP), **{pc['lime'][0]:.3f}** (LIME) and **{pc['zscore'][0]:.3f}** (z-score), "
            f"vs **{pc['random'][0]:.3f}** for a random feature. **{charts.METHOD_LABELS[best]}** is the most faithful.\n"
            f"- The z-score method picks the same top feature as SHAP for only "
            f"**{d['agreement']['top1_match']['zscore_vs_shap']:.0%}** of reviews: it describes what is unusual "
            "about a review, not what the model actually used.")

# --- Genuine vs fake ---
with tabs[4]:
    if has_labels:
        from explanation_eval import class_contrast
        contrast = class_contrast(shap_vals, df["ground_truth_label"])
        st.plotly_chart(charts.class_contrast_bar(contrast), use_container_width=True)
        top_fake = min(contrast["fake"], key=contrast["fake"].get)
        st.markdown(f"For fake reviews, **{top_fake}** pushes credibility down the most on average.")
    else:
        st.info("Needs ground-truth labels: pick the Kaggle dataset on Home.")
