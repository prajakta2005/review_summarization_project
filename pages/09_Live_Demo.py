"""
Live Demo: type (or pick) any review and see the whole system work on it:
credibility score -> SHAP reasons -> influential words (LIME) -> plain-
English explanation -> how it would be weighted in a summary.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import plotly.graph_objects as go
import streamlit as st

from dashboard import require_pipeline, get_text_model
from single_review import score_single_review
from lime_explainer import lime_text_explain, highlight_html
from charts.explainability import shap_waterfall

st.set_page_config(page_title="Live Demo", layout="wide")
st.title("Live Demo — score any review")

res = require_pipeline()
ref = res.explained
categories = sorted(ref["product_id"].unique())
has_labels = "ground_truth_label" in ref.columns and ref["ground_truth_label"].notna().all()
test = ref.iloc[res.test_idx]

DEFAULT_TEXT = ("I've used this blender daily for 3 weeks. The motor is strong enough for frozen fruit, "
                "but the lid cracked on day 12 and the replacement took 5 days to arrive.")
st.session_state.setdefault("demo_text", DEFAULT_TEXT)
st.session_state.setdefault("demo_rating", 3)
st.session_state.setdefault("demo_category", categories[0])
st.session_state.setdefault("demo_truth", None)


def load_example(kind: str):
    """Fill the inputs from a held-out test review the model didn't train on."""
    if kind == "random":
        pool = test
    elif has_labels:
        want = 1 if kind == "genuine" else 0
        correct = (test["credibility_score"] >= 50).astype(int) == test["ground_truth_label"]
        pool = test[(test["ground_truth_label"] == want) & correct]
        # clear-cut examples read better in a demo
        pool = pool[pool["credibility_score"] >= 65] if want == 1 else pool[pool["credibility_score"] <= 35]
    else:
        pool = test[test["credibility_score"] >= 70] if kind == "genuine" else test[test["credibility_score"] < 40]
    if pool.empty:
        pool = test
    row = pool.sample(1).iloc[0]
    st.session_state["demo_text"] = row["review_text"]
    st.session_state["demo_rating"] = int(row["rating"])
    st.session_state["demo_category"] = row["product_id"]
    st.session_state["demo_truth"] = (("genuine" if row["ground_truth_label"] == 1 else "fake")
                                      if has_labels else None)


def clear_truth():
    st.session_state["demo_truth"] = None


left, right = st.columns([3, 1])
left.text_area("Review text", key="demo_text", height=130, on_change=clear_truth)
right.slider("Star rating", 1, 5, key="demo_rating", on_change=clear_truth)
right.selectbox("Category / product", categories, key="demo_category", on_change=clear_truth)

b = st.columns(4)
b[0].button("🟢 Genuine example", on_click=load_example, args=("genuine",))
b[1].button("🔴 Fake example", on_click=load_example, args=("fake",))
b[2].button("🎲 Random test review", on_click=load_example, args=("random",))
b[3].caption("Examples come from the held-out test split, so the model never saw them.")

text = st.session_state["demo_text"].strip()
if not text:
    st.info("Type a review to score it.")
    st.stop()

out = score_single_review(text, st.session_state["demo_rating"], st.session_state["demo_category"], res.clf, ref)
score = out["credibility_score"]
color = "#2ecc71" if score >= 70 else "#f1c40f" if score >= 40 else "#e74c3c"

# --- score + reasons ---
c1, c2 = st.columns([1, 2])
with c1:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": "%"},
        title={"text": f"Credibility · {out['credibility_label']}"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 40], "color": "rgba(231,76,60,0.15)"},
                         {"range": [40, 70], "color": "rgba(241,196,15,0.15)"},
                         {"range": [70, 100], "color": "rgba(46,204,113,0.15)"}]},
    ))
    gauge.update_layout(height=280, margin=dict(t=60, b=10, l=30, r=30))
    st.plotly_chart(gauge, use_container_width=True)
    st.caption(f"Model confidence {out['confidence']:.0f}%")
    truth = st.session_state.get("demo_truth")
    if truth:
        predicted = "genuine" if score >= 50 else "fake"
        (st.success if predicted == truth else st.error)(
            f"True label: **{truth}** — the model {'got it right' if predicted == truth else 'got it wrong'}.")
with c2:
    st.plotly_chart(shap_waterfall(out["shap_factors"], out["base_value"]), use_container_width=True)

pf, nf = st.columns(2)
with pf:
    st.markdown("**✅ Made it more credible**")
    for f in out["shap_explanation"]["positive_factors"] or ["(nothing notable)"]:
        st.markdown(f"- {f}")
with nf:
    st.markdown("**⚠️ Made it less credible**")
    for f in out["shap_explanation"]["negative_factors"] or ["(nothing notable)"]:
        st.markdown(f"- {f}")

st.markdown("**Plain-English explanation**")
st.info(out["shap_explanation"]["explanation_text"])

if st.checkbox("Ask Gemini for a richer explanation"):
    from llm_explainer import generate_llm_explanation_safe
    with st.spinner("Asking Gemini..."):
        st.success(generate_llm_explanation_safe(
            out["row"], out["shap_explanation"]["positive_factors"], out["shap_explanation"]["negative_factors"],
            fallback_text=out["shap_explanation"]["explanation_text"], shap_factors=out["shap_factors"]))

# --- words ---
st.markdown("**Words that mattered** (LIME on the TF-IDF text model: green → genuine, red → fake)")
words = lime_text_explain(get_text_model(res), text)
st.markdown(highlight_html(text, words), unsafe_allow_html=True)

# --- features + summary placement ---
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown("**Feature values vs the dataset**")
    st.dataframe(out["feature_comparison"], use_container_width=True)
with c2:
    st.markdown("**Where would this review land in a summary?**")
    pct = out["category_percentile"]
    st.markdown(
        f"In a **credibility-aware** summary of *{st.session_state['demo_category']}*, this review's sentences "
        f"get weight **{out['summary_weight']:.2f}**"
        + (f", more than **{pct:.0%}** of reviews in that category." if pct is not None else ".")
        + " In a **standard** summary every review gets the same weight (1.0), however credible it is.")
    if score < 40:
        st.warning("Its score is below 40, so it would also be listed under the summary's "
                   "low-credibility flags (surfaced, not deleted).")
    st.page_link("pages/06_Summarization.py", label="See the summaries →")
