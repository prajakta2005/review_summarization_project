"""
Ablation charts. Builders take the per-model dict from
results/ablation.json -> data.models.<rf|lr>.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from features import FEATURE_COLUMNS

METRIC_LABELS = {"f1": "F1", "accuracy": "Accuracy", "roc_auc": "ROC-AUC"}


def delta_bar(loo: list, full: dict, metric: str = "f1") -> go.Figure:
    """Change in the metric when each feature is removed (red = big drop)."""
    rows = [{"removed": r["removed"], "delta": r[metric]["mean"] - full[metric]["mean"],
             "std": r[metric]["std"]} for r in loo]
    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    fig = px.bar(df, x="delta", y="removed", orientation="h", error_x="std",
                 color="delta", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                 text="delta",
                 title=f"Change in {METRIC_LABELS[metric]} when one feature is removed")
    fig.update_traces(texttemplate="%{text:+.4f}", textposition="outside", cliponaxis=False)
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    fig.update_layout(yaxis_title="feature removed", xaxis_title=f"Δ {METRIC_LABELS[metric]} (error bar = fold std)",
                      coloraxis_showscale=False)
    return fig


def single_feature_bar(single: list, metric: str = "f1") -> go.Figure:
    df = pd.DataFrame([{"feature": r["feature"], "score": r[metric]["mean"], "std": r[metric]["std"]}
                       for r in single]).sort_values("score")
    fig = px.bar(df, x="score", y="feature", orientation="h", error_x="std", text="score",
                 title=f"{METRIC_LABELS[metric]} using ONE feature only")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False,
                      marker_color="#3498db")
    chance = 0.5
    fig.add_vline(x=chance, line_dash="dash", line_color="grey", annotation_text="chance")
    fig.update_layout(yaxis_title=None, xaxis_range=[min(0.3, df["score"].min() - 0.05), 1])
    return fig


def forward_curve(forward: list, full: dict, metric: str = "f1") -> go.Figure:
    df = pd.DataFrame([{"n_features": r["step"], "added": r["added"], "score": r[metric]["mean"],
                        "std": r[metric]["std"]} for r in forward])
    fig = go.Figure()
    mean, std = full[metric]["mean"], full[metric]["std"]
    fig.add_hrect(y0=mean - std, y1=mean + std, fillcolor="lightgrey", opacity=0.4, line_width=0,
                  annotation_text="all 6 features ± std", annotation_position="bottom right")
    fig.add_scatter(x=df["n_features"], y=df["score"], mode="lines+markers+text",
                    text=["+ " + a for a in df["added"]], textposition="top left",
                    error_y=dict(type="data", array=df["std"]), line=dict(color="#2c3e50"),
                    name=METRIC_LABELS[metric])
    fig.update_layout(title=f"{METRIC_LABELS[metric]} as features are added (greedy, best first)",
                      xaxis_title="number of features", yaxis_title=METRIC_LABELS[metric],
                      xaxis=dict(tickmode="linear", dtick=1), showlegend=False)
    return fig


def group_bar(groups: list, full: dict, metric: str = "f1") -> go.Figure:
    df = pd.DataFrame([{"group": f"{r['removed_group']} ({', '.join(r['features'])})",
                        "delta": r[metric]["mean"] - full[metric]["mean"], "std": r[metric]["std"]}
                       for r in groups]).sort_values("delta", ascending=False)
    fig = px.bar(df, x="delta", y="group", orientation="h", error_x="std", text="delta",
                 color="delta", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                 title=f"Change in {METRIC_LABELS[metric]} when a feature GROUP is removed")
    fig.update_traces(texttemplate="%{text:+.4f}", textposition="outside", cliponaxis=False)
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    fig.update_layout(yaxis_title=None, coloraxis_showscale=False)
    return fig


def ranking_table(rankings: dict) -> pd.DataFrame:
    """feature x method -> rank (1 = most important)."""
    methods = [m for m in ["ablation_loo", "permutation", "impurity", "shap"] if m in rankings]
    table = pd.DataFrame(index=FEATURE_COLUMNS)
    for m in methods:
        table[m] = [rankings[m].index(f) + 1 for f in FEATURE_COLUMNS]
    return table.sort_values("ablation_loo").rename(columns={
        "ablation_loo": "Ablation (leave-one-out)", "permutation": "Permutation",
        "impurity": "RF impurity", "shap": "Mean |SHAP|"})
