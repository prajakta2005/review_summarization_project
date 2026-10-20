"""
Summarization: standard vs credibility-aware summaries, side by side,
with every sentence traced to its review, credibility score and true
label, plus the full-dataset evaluation (fake contamination, ROUGE).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import html

import pandas as pd
import streamlit as st

from dashboard import require_pipeline, pipeline_key, show_result_or_hint
from summary_eval import evaluate_summaries
from charts import summarization as charts
from charts.summarization import METRIC_LABELS, VARIANT_LABELS

st.set_page_config(page_title="Summarization", layout="wide")
st.title("Summarization: Standard vs Credibility-Aware")

res = require_pipeline()
df = res.explained
has_labels = "ground_truth_label" in df.columns and df["ground_truth_label"].notna().all()


@st.cache_data(show_spinner="Building summaries...")
def live_eval(key, product_id, top_n, weight_floor, hard_threshold, _df):
    return evaluate_summaries(_df, product_id, top_n, weight_floor=weight_floor, hard_threshold=hard_threshold)


# --- controls ---
c = st.columns([2, 1, 1, 1])
product = c[0].selectbox("Category / product", sorted(df["product_id"].unique()))
top_n = c[1].radio("Sentences", [3, 5, 8], index=1, horizontal=True)
floor = c[2].slider("Weight floor", 0.0, 0.5, 0.05, 0.05,
                    help="Minimum weight for a sentence, so low-credibility text is down-weighted but never zeroed.")
hard = c[3].checkbox("Drop reviews < 40", help="Hard filter instead of soft weighting, for comparison.")
result = live_eval(pipeline_key() + (len(df),), product, top_n, floor, 40 if hard else None, df)
m_std, m_ours = result["metrics"]["standard"], result["metrics"]["credibility_aware"]

# --- headline deltas ---
k = st.columns(3)
if has_labels:
    k[0].metric("Fake contamination", f"{m_ours['contamination']:.0%}",
                f"{(m_ours['contamination'] - m_std['contamination']) * 100:+.0f} pp vs standard ({m_std['contamination']:.0%})",
                delta_color="inverse")
    k[2].metric("ROUGE-L vs genuine-only summary", f"{m_ours['rougeL']:.3f}",
                f"{m_ours['rougeL'] - m_std['rougeL']:+.3f} vs standard ({m_std['rougeL']:.3f})")
else:
    k[0].metric("Fake contamination", "n/a", help="Needs ground-truth labels (Kaggle dataset).")
    k[2].metric("ROUGE-L vs genuine-only summary", "n/a")
k[1].metric("Mean source credibility", f"{m_ours['mean_source_cred']:.1f}",
            f"{m_ours['mean_source_cred'] - m_std['mean_source_cred']:+.1f} vs standard ({m_std['mean_source_cred']:.1f})")


def sentence_card(item: dict) -> str:
    score = item["score"]
    color = "#2ecc71" if score >= 70 else "#f1c40f" if score >= 40 else "#e74c3c"
    if item.get("is_fake") is None:
        badge = ""
    elif item["is_fake"]:
        badge = '<span style="background:#e74c3c;color:white;border-radius:4px;padding:0 6px;font-size:0.75em">FAKE</span>'
    else:
        badge = '<span style="background:#2ecc71;color:white;border-radius:4px;padding:0 6px;font-size:0.75em">GENUINE</span>'
    return (f'<div style="border-left:5px solid {color};padding:6px 10px;margin:6px 0;'
            f'background:rgba(127,127,127,0.06);border-radius:4px">'
            f'{html.escape(item["sentence"])}<br>'
            f'<span style="font-size:0.8em;opacity:0.75">review #{item["review_id"]} · credibility {score:.0f}%</span> {badge}</div>')


left, right = st.columns(2)
for col, variant in [(left, "standard"), (right, "credibility_aware")]:
    col.subheader(VARIANT_LABELS[variant])
    col.markdown("".join(sentence_card(it) for it in result["summaries"][variant]), unsafe_allow_html=True)
if has_labels:
    with st.expander("Genuine-only (oracle) summary: the reference the ROUGE scores compare against"):
        st.markdown("".join(sentence_card(it) for it in result["summaries"]["oracle"]), unsafe_allow_html=True)
st.caption("Left border = the source review's credibility (green ≥ 70, yellow 40–70, red < 40); badge = ground-truth label. "
           "Live scores come from the model trained on this dataset (some of these reviews were in its training "
           "split); the full-dataset results below use out-of-fold scores, so no review is scored by a model that saw it.")

# --- full-dataset results ---
tabs = st.tabs(["All categories", "Sensitivity", "Method"])
saved = show_result_or_hint("summary_comparison", "experiments/run_summary_comparison.py")
if saved:
    d = saved["data"]
    with tabs[0]:
        n = st.radio("Summary length", d["settings"]["top_n"], index=d["settings"]["top_n"].index(5)
                     if 5 in d["settings"]["top_n"] else 0, horizontal=True, key="saved_topn")
        avg = d["averages"][str(n)]
        st.markdown(f"Averages over **{d['n_categories']} categories**, {saved['meta']['n_rows']:,} reviews, "
                    "out-of-fold credibility scores.")
        st.dataframe(charts.metric_delta_table(avg), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.contamination_bar(d["per_category"], n), use_container_width=True)
        c2.plotly_chart(charts.rouge_grouped_bar(avg), use_container_width=True)
        st.plotly_chart(charts.topn_line(d["averages"]), use_container_width=True)
    with tabs[1]:
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.sensitivity_line(d["sensitivity"]["hard_threshold"], "hard_threshold"),
                        use_container_width=True)
        c2.plotly_chart(charts.sensitivity_line(d["sensitivity"]["weight_floor"], "weight_floor"),
                        use_container_width=True)
        ht = pd.DataFrame(d["sensitivity"]["hard_threshold"])
        st.markdown(
            "- **Hard filtering** trades content for purity: dropping more low-credibility reviews keeps cutting "
            f"contamination (to {ht['contamination'].iloc[-1]:.0%} at {ht['hard_threshold'].iloc[-1]}) but past a point "
            "it also throws away genuine sentences and ROUGE-L falls.\n"
            "- **The weight floor** barely matters at these settings: the sentences that win the ranking come from "
            "reviews well above any floor tested.")
    with tabs[2]:
        st.markdown("""
All three summaries use the **same extractive ranker** (TF-IDF similarity to the centroid of all sentences),
so the only difference is how much each review is trusted:

| Summary | Sentences from | Weight |
|---|---|---|
| **Standard** | all reviews | 1.0 for every sentence |
| **Credibility-aware (ours)** | all reviews | the review's credibility score / 100 (minimum = weight floor) |
| **Genuine-only (oracle)** | ground-truth genuine reviews only | 1.0 |

**Metrics.** *Fake contamination* is the share of summary sentences quoted from ground-truth fake (computer-generated)
reviews. *ROUGE* measures overlap with the oracle summary. *Redundancy* is the mean similarity between summary sentences.
*Coverage* is the share of the category's three most-discussed attributes that the summary mentions.

**Leakage control.** In the saved results every review's credibility score comes from 5-fold out-of-fold
predictions, so no review is scored by a model that was trained on it.

**Why the standard summary is so contaminated.** Computer-generated reviews use generic, typical wording, which sits
close to the centroid the ranker rewards. That's exactly what makes an un-weighted summary echo the fakes.
""")
