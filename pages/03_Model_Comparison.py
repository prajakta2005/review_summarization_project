"""
Model Comparison: Random Forest vs Logistic Regression vs SVM vs
DistilBERT, on handcrafted / TF-IDF / hybrid features, all evaluated on
the same held-out test set. Reads results/model_comparison.json.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd

from dashboard import show_result_or_hint
from charts import model_comparison as charts
from charts.model_comparison import METRIC_LABELS

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison")

result = show_result_or_hint("model_comparison", "experiments/run_model_comparison.py")
if result is None:
    st.stop()

meta, data = result["meta"], result["data"]
st.caption(f"{meta['dataset']} · {meta['n_rows']:,} reviews · test set {data['n_test']:,} reviews "
           f"(same split for every model) · generated {meta['generated_at']} · commit {meta['git_commit']}")

all_models = data["models"]
by_id = {m["model_id"]: m for m in all_models}

# --- filters ---
f1, f2 = st.columns([3, 1])
feature_sets = sorted({m["feature_set"] for m in all_models},
                      key=["handcrafted", "tfidf", "hybrid", "bert"].index)
chosen_sets = f1.multiselect("Feature sets", feature_sets, default=feature_sets)
metric = f2.selectbox("Metric", ["f1", "accuracy", "precision", "recall", "roc_auc", "pr_auc"],
                      format_func=METRIC_LABELS.get)
models = [m for m in all_models if m["feature_set"] in chosen_sets]
if not models:
    st.warning("Pick at least one feature set.")
    st.stop()

# --- headline cards ---
best = max(models, key=lambda m: m[metric] or 0)
classical = [m for m in models if m["feature_set"] != "bert"]
best_classical = max(classical, key=lambda m: m[metric] or 0) if classical else None
deployed = by_id.get(data["deployed_model"])

c = st.columns(3)
c[0].metric(f"🏆 Best {METRIC_LABELS[metric]}", f"{best[metric]:.3f}", help=f"{best['model']} · {best['feature_set']}")
c[0].caption(f"{best['model']} · {best['feature_set']}")
if best_classical:
    c[1].metric(f"Best classical {METRIC_LABELS[metric]}", f"{best_classical[metric]:.3f}")
    c[1].caption(f"{best_classical['model']} · {best_classical['feature_set']}")
if deployed:
    c[2].metric(f"Deployed model {METRIC_LABELS[metric]}", f"{deployed[metric]:.3f}",
                delta=f"{deployed[metric] - best[metric]:+.3f} vs best", delta_color="off")
    c[2].caption("Random Forest · 6 handcrafted features (what the dashboard runs)")

if "distilbert" not in by_id:
    st.info("DistilBERT isn't in the results yet. Run `notebooks/bert_finetune_colab.ipynb` on Google Colab, "
            "save its two output files to `results/predictions/`, then run "
            "`python experiments/run_model_comparison.py --only distilbert`.")

st.plotly_chart(charts.leaderboard_bar(models, metric, data["deployed_model"]), use_container_width=True)

left, right = st.columns(2)
left.plotly_chart(charts.metric_heatmap(models), use_container_width=True)
right.plotly_chart(charts.cost_vs_quality_scatter(models, metric), use_container_width=True)

st.plotly_chart(charts.featureset_grouped_bar(models, metric), use_container_width=True)

# --- interpretation ---
hand = [m for m in all_models if m["feature_set"] == "handcrafted"]
text = [m for m in all_models if m["feature_set"] in ("tfidf", "hybrid")]
if hand and text:
    best_hand = max(hand, key=lambda m: m["f1"])
    best_text = max(text, key=lambda m: m["f1"])
    lr_t, lr_h = by_id.get("lr_tfidf"), by_id.get("lr_hybrid")
    lines = [
        f"- Text models beat the 6 interpretable features: best text F1 **{best_text['f1']:.3f}** "
        f"({best_text['model']} · {best_text['feature_set']}) vs best handcrafted **{best_hand['f1']:.3f}** "
        f"({best_hand['model']}).",
    ]
    if lr_t and lr_h:
        diff = lr_h["f1"] - lr_t["f1"]
        verdict = "add a little" if diff > 0.001 else ("add nothing measurable" if abs(diff) <= 0.001 else "slightly hurt")
        lines.append(f"- Adding the handcrafted features to TF-IDF (LR hybrid vs LR tfidf) changes F1 by "
                     f"**{diff:+.4f}**: the credibility signals {verdict} on top of the words.")
    st.subheader("What this shows")
    st.markdown("\n".join(lines))
st.info("**Why the dashboard still uses the Random Forest on 6 handcrafted features:** every score it gives "
        "can be explained in terms of six human-readable signals (length, specificity, repetition, ...), "
        "and those explanations feed the credibility-aware summaries. Text and BERT models score higher "
        "but can't give reasons at that level. That trade-off is a deliberate design choice.")

with st.expander("Full results table"):
    cols = ["model", "feature_set", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
            "train_seconds", "predict_seconds", "n_train", "n_test"]
    table = pd.DataFrame(all_models)[cols]
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", table.to_csv(index=False), "model_comparison.csv", "text/csv")

with st.expander("How to regenerate these results"):
    st.code("python experiments/run_dataset_profile.py      # writes the shared split\n"
            "python experiments/run_model_comparison.py     # ~3 min for the 9 classical models\n"
            "# DistilBERT: run notebooks/bert_finetune_colab.ipynb on Colab (T4 GPU), save\n"
            "# distilbert.csv + distilbert_meta.json to results/predictions/, then:\n"
            "python experiments/run_model_comparison.py --only distilbert", language="bash")
