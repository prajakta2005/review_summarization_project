"""
Ablation Study: retrain without (or with only) some credibility features
to measure what each one contributes. Reads results/ablation.json.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd

from dashboard import show_result_or_hint
from charts import ablation as charts
from charts.ablation import METRIC_LABELS

st.set_page_config(page_title="Ablation Study", layout="wide")
st.title("Ablation Study")

result = show_result_or_hint("ablation", "experiments/run_ablation.py")
if result is None:
    st.stop()
meta, data = result["meta"], result["data"]
st.caption(f"{data['k']}-fold stratified cross-validation on {meta['n_rows']:,} reviews ({meta['dataset']}) "
           f"· generated {meta['generated_at']} · commit {meta['git_commit']}")

c1, c2 = st.columns(2)
model_key = c1.radio("Model", list(data["models"]), horizontal=True,
                     format_func={"rf": "Random Forest (deployed)", "lr": "Logistic Regression"}.get)
metric = c2.selectbox("Metric", ["f1", "accuracy", "roc_auc"], format_func=METRIC_LABELS.get)
res = data["models"][model_key]
full = res["full"]
label = METRIC_LABELS[metric]


def drop(r):
    return r[metric]["mean"] - full[metric]["mean"]


loo = sorted(res["leave_one_out"], key=drop)
most, least = loo[0], loo[-1]
target = full[metric]["mean"] * 0.99
needed = next((r["step"] for r in res["forward"] if r[metric]["mean"] >= target), len(res["forward"]))

k = st.columns(4)
k[0].metric(f"All 6 features · {label}", f"{full[metric]['mean']:.4f}", help=f"± {full[metric]['std']:.4f} across folds")
k[1].metric("Most important (removal hurts most)", most["removed"], f"{drop(most):+.4f}", delta_color="off")
k[2].metric("Least important", least["removed"], f"{drop(least):+.4f}", delta_color="off")
k[3].metric(f"Features needed for 99% of {label}", f"{needed} of 6")

tabs = st.tabs(["Leave-one-out", "Single feature", "Forward addition", "Feature groups", "Rankings", "Text vs features"])

with tabs[0]:
    st.plotly_chart(charts.delta_bar(res["leave_one_out"], full, metric), use_container_width=True)
    st.markdown(f"Removing **{most['removed']}** costs the most {label} "
                f"({drop(most):+.4f} ± {most[metric]['std']:.4f}). "
                + (f"Removing **{least['removed']}** {'improves' if drop(least) > 0 else 'barely changes'} "
                   f"{label} ({drop(least):+.4f}), so the model gets little from it."))
    st.dataframe(pd.DataFrame([{"removed": r["removed"], label: r[metric]["mean"], "std": r[metric]["std"],
                                f"Δ {label}": round(drop(r), 4)} for r in loo]),
                 use_container_width=True, hide_index=True)

with tabs[1]:
    st.plotly_chart(charts.single_feature_bar(res["single_feature"], metric), use_container_width=True)
    best_single = max(res["single_feature"], key=lambda r: r[metric]["mean"])
    st.markdown(f"On its own, **{best_single['feature']}** reaches {label} "
                f"{best_single[metric]['mean']:.4f}, vs {full[metric]['mean']:.4f} with all six features.")

with tabs[2]:
    st.plotly_chart(charts.forward_curve(res["forward"], full, metric), use_container_width=True)
    order = " → ".join(r["added"] for r in res["forward"])
    st.markdown(f"Greedy order: {order}. **{needed}** feature(s) reach 99% of the full model's {label}.")

with tabs[3]:
    st.plotly_chart(charts.group_bar(res["groups"], full, metric), use_container_width=True)
    worst_group = min(res["groups"], key=drop)
    st.markdown(f"Losing the **{worst_group['removed_group']}** group "
                f"({', '.join(worst_group['features'])}) hurts most ({drop(worst_group):+.4f}).")

with tabs[4]:
    rankings = data["rankings"]
    st.markdown("Do the cheap importance measures agree with the ablation (the most direct measure)? "
                "Rank 1 = most important. Ablation and permutation are measured on F1 with the Random Forest.")
    st.dataframe(charts.ranking_table(rankings), use_container_width=True)
    rho = rankings["spearman_vs_ablation"]
    r = st.columns(3)
    r[0].metric("Spearman ρ: permutation vs ablation", rho.get("permutation"))
    r[1].metric("Spearman ρ: RF impurity vs ablation", rho.get("impurity"))
    r[2].metric("Spearman ρ: SHAP vs ablation", rho.get("shap") if "shap" in rho else "pending")
    if "shap" not in rho:
        st.caption("SHAP ranking appears after feature 05 (explainability) writes "
                   "`results/explainability_eval.json` and the ablation script is re-run.")
    if rho.get("impurity") is not None and rho["impurity"] < 0.8:
        st.warning("RF impurity importance (the chart on Home) disagrees noticeably with the ablation. "
                   "Impurity importance is biased toward features with many distinct values, so prefer the ablation "
                   "or permutation ranking when explaining which signals matter.")

with tabs[5]:
    rep = data.get("representation")
    if not rep:
        st.info("Run `python experiments/run_model_comparison.py` first, then re-run the ablation script.")
    else:
        st.markdown("Does adding the 6 handcrafted features to a TF-IDF text model help? (from the model comparison)")
        table = pd.DataFrame(rep).T.rename_axis("model family").reset_index()
        st.dataframe(table, use_container_width=True, hide_index=True)
        if all(v.get("p_value", 0) >= 0.05 for v in rep.values()):
            st.markdown("The differences are **not statistically significant** (McNemar p ≥ 0.05): on top of the "
                        "words, the handcrafted signals add no measurable accuracy. Their value in this system is "
                        "**explainability**, not extra accuracy.")
