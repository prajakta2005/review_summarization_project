"""
Novelty: what's new in combining credibility checking, explainability
and summarization -- the pipeline, the evidence from our experiments,
a capability comparison, and a live trace from a summary sentence back
to the review, its score and the reasons for it.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st

from dashboard import require_pipeline, get_shap_values
from novelty import capability_matrix, evidence_cards, standard_rank, LIMITATIONS
from summarizer import credibility_aware_summary
from shap_explainer import shap_row_factors, base_value
from charts.explainability import shap_waterfall

st.set_page_config(page_title="Novelty", layout="wide")
st.title("What's new in this system")

st.markdown(
    "Fake-review detection, explainable AI and review summarization are each well studied. "
    "This system **couples** them: the credibility model decides how much each review counts in the summary, "
    "every summary sentence can be traced back to its review, score and reasons, and we **measure** how much "
    "fake content the coupling keeps out of the summary.")

st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f4f6f7", fontname="sans-serif"];
  edge [fontname="sans-serif", fontsize=10];
  reviews [label="Reviews"]; prep [label="Preprocess"]; feat [label="6 credibility\\nfeatures"];
  model [label="Credibility model\\n(Random Forest)", fillcolor="#d6eaf8"];
  explain [label="Explanations\\n(SHAP / LIME)", fillcolor="#d5f5e3"];
  summary [label="Credibility-aware\\nsummary", fillcolor="#fdebd0"];
  flags [label="Low-credibility\\nflags (surfaced)"];
  reviews -> prep -> feat -> model;
  model -> explain [label="why this score"];
  model -> summary [label="weights each review", color="#e67e22", penwidth=3, fontcolor="#e67e22"];
  model -> flags [label="score < 40"];
  summary -> explain [label="trace each sentence", style=dashed, dir=back];
}
""")
st.caption("The orange edge is the novel coupling: credibility scores directly change what goes into the summary.")

# --- evidence ---
st.subheader("Evidence from our experiments")
cards = evidence_cards()
cols = st.columns(len(cards))
for col, card in zip(cols, cards):
    col.metric(card["label"], card["value"])
    col.caption(card["detail"])

st.subheader("Capability comparison")
st.dataframe(capability_matrix(), use_container_width=True)
st.caption("Prior-work rows are families of approaches with representative references (see docs/NOVELTY.md). "
           "Check the closest combined works in the literature before claiming novelty in the report.")

# --- trace a sentence ---
st.subheader("Trace a summary sentence")
res = require_pipeline()
df = res.explained
category = st.selectbox("Category / product", sorted(df["product_id"].unique()))
summary = credibility_aware_summary(df, category, top_n=5)
items = summary["overall_items"]
if not items:
    st.info("No summary for this category.")
    st.stop()
choice = st.radio("Credibility-aware summary sentences — pick one to trace",
                  list(range(len(items))), format_func=lambda i: items[i]["sentence"][:140])
item = items[choice]
pos = df.index[df["review_id"] == item["review_id"]][0]
i = df.index.get_loc(pos)
row = df.iloc[i]

c1, c2 = st.columns([2, 3])
with c1:
    st.markdown(f"**Source review #{row['review_id']}** · {int(row['rating'])}★")
    st.markdown(f"> {row['review_text'][:700]}")
    m = st.columns(2)
    m[0].metric("Credibility", f"{row['credibility_score']:.0f}%", row["credibility_label"], delta_color="off")
    if "ground_truth_label" in df.columns and df["ground_truth_label"].notna().all():
        m[1].metric("True label", "genuine" if row["ground_truth_label"] == 1 else "fake")
    rank, n = standard_rank(df, category, item["review_id"], item["sentence"])
    if rank:
        st.markdown(f"In the **standard** (unweighted) summary this sentence would rank **#{rank}** of {n:,} "
                    f"sentences{' — it would be included there too' if rank <= 5 else ''}.")
with c2:
    shap_row = get_shap_values(res)[i]
    st.plotly_chart(shap_waterfall(shap_row_factors(shap_row, row), base_value(res.clf)),
                    use_container_width=True)

st.subheader("Limitations")
st.markdown("\n".join(f"- {l}" for l in LIMITATIONS))
