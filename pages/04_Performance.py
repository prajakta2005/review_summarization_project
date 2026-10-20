"""
Performance Results: accuracy, precision, recall, F1, ROC/PR-AUC and the
confusion matrix for every model, plus per-class and per-category
breakdowns, a live decision-threshold explorer, 5-fold cross-validation
and McNemar significance tests. Reads results/ only -- no retraining.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd

from dashboard import show_result_or_hint
from pipeline import load_raw
from performance import (load_predictions, per_class_report, per_category_metrics,
                         threshold_sweep, mcnemar)
from evaluation import compute_metrics
from credibility_model import HIGH_THRESHOLD, LOW_THRESHOLD
from charts import performance as charts
from charts.model_comparison import METRIC_LABELS

st.set_page_config(page_title="Performance", layout="wide")
st.title("Performance Results")

comparison = show_result_or_hint("model_comparison", "experiments/run_model_comparison.py")
if comparison is None:
    st.stop()
data = comparison["data"]
models = data["models"]
by_id = {m["model_id"]: m for m in models}
label_of = {m["model_id"]: f"{m['model']} · {m['feature_set']}" for m in models}
deployed_id = data["deployed_model"]

st.caption(f"Test set: {data['n_test']:,} held-out reviews from {comparison['meta']['dataset']} "
           f"(identical for every model) · generated {comparison['meta']['generated_at']}")


@st.cache_data(show_spinner="Loading review texts...")
def review_lookup() -> pd.DataFrame:
    raw = load_raw("kaggle_full")
    return raw.set_index("review_id")[["category", "rating", "review_text"]]


# --- model picker + headline metrics ---
ids = list(by_id)
p1, p2 = st.columns([2, 2])
model_id = p1.selectbox("Model", ids, index=ids.index(deployed_id), format_func=label_of.get)
compare_id = p2.selectbox("Compare against", ids, index=ids.index(deployed_id) if model_id != deployed_id
                          else ids.index(data["best_by_f1"]), format_func=label_of.get)
m, ref = by_id[model_id], by_id[compare_id]

cards = st.columns(6)
for col, key in zip(cards, ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]):
    delta = None if model_id == compare_id or m[key] is None else f"{m[key] - ref[key]:+.4f}"
    col.metric(METRIC_LABELS[key], f"{m[key]:.4f}" if m[key] is not None else "n/a", delta=delta)
if model_id != compare_id:
    st.caption(f"Deltas are vs **{label_of[compare_id]}**. Precision/recall treat *genuine* as the positive class.")

tabs = st.tabs(["Confusion Matrix", "Curves", "Threshold", "By Category", "Cross-Validation", "Significance"])
preds = load_predictions(model_id)

# --- Confusion matrix ---
with tabs[0]:
    normalize = st.toggle("Normalize by row (show recall per class)", value=False)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(charts.confusion_matrix_heatmap(m["confusion_matrix"], normalize,
                                                    title=f"Confusion matrix — {label_of[model_id]}"),
                    use_container_width=True)
    if preds is not None:
        c2.markdown("**Per-class metrics**")
        c2.dataframe(per_class_report(preds["y_true"], preds["y_pred"]), use_container_width=True)
        (tn, fp), (fn, tp) = m["confusion_matrix"]
        c2.markdown(f"- **{fp:,}** fake reviews were judged genuine (false positives)\n"
                    f"- **{fn:,}** genuine reviews were judged fake (false negatives)")

        st.markdown("**Example mistakes**")
        texts = review_lookup()
        e1, e2 = st.columns(2)
        for col, (title, mask) in zip([e1, e2], [
            ("Fake, but predicted genuine", (preds["y_true"] == 0) & (preds["y_pred"] == 1)),
            ("Genuine, but predicted fake", (preds["y_true"] == 1) & (preds["y_pred"] == 0)),
        ]):
            col.markdown(f"*{title}*")
            for _, r in preds[mask].sample(min(3, int(mask.sum())), random_state=1).iterrows():
                row = texts.loc[r["review_id"]]
                col.info(f"**#{r['review_id']}** · {row['category']} · {int(row['rating'])}★ · "
                         f"score {r['score']:.2f}\n\n{str(row['review_text'])[:300]}")

# --- ROC / PR curves ---
with tabs[1]:
    chosen = st.multiselect("Models to overlay", ids, default=ids, format_func=label_of.get)
    shown = [by_id[i] for i in chosen]
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.roc_curves(shown), use_container_width=True)
    c2.plotly_chart(charts.pr_curves(shown), use_container_width=True)

# --- Threshold explorer ---
with tabs[2]:
    if preds is None:
        st.info("No saved predictions for this model.")
    else:
        sweep = threshold_sweep(preds["y_true"], preds["score"])
        is_prob = m.get("score_type", "probability") == "probability"
        lo, hi = float(preds["score"].min()), float(preds["score"].max())
        default = 0.5 if is_prob else 0.0
        t = st.slider("Decision threshold (score ≥ threshold → genuine)",
                      0.0 if is_prob else round(lo, 2), 1.0 if is_prob else round(hi, 2),
                      default, 0.01)
        markers = {"Low cut-off": LOW_THRESHOLD / 100, "High cut-off": HIGH_THRESHOLD / 100} if is_prob else {}
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(charts.threshold_curve(sweep, t, markers), use_container_width=True)
        y_pred_t = (preds["score"] >= t).astype(int)
        live = compute_metrics(preds["y_true"], y_pred_t, curves=False)
        c2.plotly_chart(charts.confusion_matrix_heatmap(live["confusion_matrix"],
                                                        title=f"At threshold {t:.2f}"),
                        use_container_width=True)
        k = c2.columns(4)
        for col, key in zip(k, ["accuracy", "precision", "recall", "f1"]):
            col.metric(METRIC_LABELS[key], f"{live[key]:.3f}")
        best = sweep.loc[sweep["f1"].idxmax()]
        st.caption(f"Best F1 for this model is {best['f1']:.3f} at threshold {best['threshold']:.2f}. "
                   + (f"The dashboard labels scores ≥ {HIGH_THRESHOLD} as High and < {LOW_THRESHOLD} as Low "
                      "credibility (dotted lines); reviews in between are Medium." if is_prob else
                      "This model outputs an SVM margin rather than a probability, so 0 is its default threshold."))

# --- Per category ---
with tabs[3]:
    if preds is not None:
        cat = per_category_metrics(preds, review_lookup()["category"])
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(charts.per_category_bar(cat), use_container_width=True)
        c2.dataframe(cat, use_container_width=True, hide_index=True)
        worst, best_cat = cat.iloc[0], cat.iloc[-1]
        st.caption(f"Hardest category: **{worst['category']}** (F1 {worst['f1']:.3f}); "
                   f"easiest: **{best_cat['category']}** (F1 {best_cat['f1']:.3f}).")

# --- Cross-validation ---
with tabs[4]:
    cv = show_result_or_hint("cv_results", "experiments/run_cross_validation.py")
    if cv:
        cv_models = cv["data"]["models"]
        metric = st.selectbox("Metric", ["f1", "accuracy", "precision", "recall", "roc_auc"],
                              format_func=METRIC_LABELS.get, key="cv_metric")
        st.plotly_chart(charts.cv_box(cv_models, metric), use_container_width=True)
        table = pd.DataFrame([
            {"model": f"{v['model']} · {v['feature_set']}",
             **{METRIC_LABELS[k]: f"{pd.Series(v[k]).mean():.4f} ± {pd.Series(v[k]).std(ddof=0):.4f}"
                for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}}
            for v in cv_models.values()
        ])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(f"{cv['data']['k']}-fold stratified CV on {cv['meta']['n_rows']:,} reviews (mean ± std). "
                   "DistilBERT is not cross-validated (5 GPU fine-tuning runs); its numbers are from the single shared split.")

# --- Significance ---
with tabs[5]:
    sig = show_result_or_hint("significance", "experiments/run_significance.py")
    if sig:
        pairs = pd.DataFrame(sig["data"]["pairs"])
        involving = pairs[(pairs["model_a"] == model_id) | (pairs["model_b"] == model_id)].copy()
        involving["other model"] = involving.apply(
            lambda r: label_of.get(r["model_b"] if r["model_a"] == model_id else r["model_a"]), axis=1)
        involving["this model wins"] = involving.apply(
            lambda r: (r["accuracy_a"] > r["accuracy_b"]) == (r["model_a"] == model_id), axis=1)
        st.markdown(f"McNemar's test: **{label_of[model_id]}** vs every other model on the same "
                    f"{sig['data']['n_test']:,} test reviews (α = {sig['data']['alpha']}).")
        st.dataframe(involving[["other model", "this model wins", "b", "c", "statistic", "p_value",
                                "significant", "method"]].sort_values("p_value"),
                     use_container_width=True, hide_index=True)
        ns = pairs[~pairs["significant"]]
        if len(ns):
            st.warning("Differences that are **not** statistically significant: " + "; ".join(
                f"{label_of.get(r.model_a)} vs {label_of.get(r.model_b)} (p = {r.p_value:.3f})"
                for r in ns.itertuples()))
        st.caption("b = reviews only this pair's first model got right, c = only the second got right.")
