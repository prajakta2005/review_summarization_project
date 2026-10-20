"""
Explainability charts: global SHAP importance, beeswarm, per-review
waterfall, LIME weights, method agreement and faithfulness.
Green = pushes toward genuine/credible, red = toward fake.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from features import FEATURE_COLUMNS

GENUINE, FAKE = "#2ecc71", "#e74c3c"
METHOD_LABELS = {"zscore": "z-score", "shap": "SHAP", "lime": "LIME", "random": "random (baseline)"}


def shap_bar(global_shap: dict) -> go.Figure:
    s = pd.Series(global_shap).sort_values()
    fig = px.bar(x=s.values, y=s.index, orientation="h", text=s.values,
                 title="Global importance: mean |SHAP| (average push on credibility)")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False,
                      marker_color="#3498db")
    fig.update_layout(xaxis_title="mean |SHAP value| (probability points)", yaxis_title=None)
    return fig


def shap_beeswarm(shap_vals: np.ndarray, X: pd.DataFrame, max_points: int = 2000, seed: int = 42) -> go.Figure:
    """One dot per review per feature: x = SHAP value, colour = the
    feature's value (blue low -> red high), like shap.summary_plot."""
    n = len(X)
    idx = np.random.default_rng(seed).choice(n, size=min(n, max_points), replace=False)
    order = list(pd.Series(np.abs(shap_vals).mean(axis=0), index=FEATURE_COLUMNS).sort_values().index)
    fig = go.Figure()
    for f in order:
        j = FEATURE_COLUMNS.index(f)
        vals = X[f].to_numpy()[idx]
        lo, hi = np.percentile(vals, [5, 95])
        color = np.clip((vals - lo) / ((hi - lo) or 1), 0, 1)
        jitter = np.random.default_rng(j).uniform(-0.3, 0.3, len(idx))
        fig.add_scatter(x=shap_vals[idx, j], y=np.full(len(idx), order.index(f)) + jitter, mode="markers",
                        marker=dict(size=4, color=color, colorscale="RdBu_r", cmin=0, cmax=1, opacity=0.6,
                                    colorbar=dict(title="feature<br>value", tickvals=[0, 1],
                                                  ticktext=["low", "high"]) if f == order[-1] else None,
                                    showscale=f == order[-1]),
                        name=f, showlegend=False,
                        hovertemplate=f"{f}<br>SHAP %{{x:.3f}}<extra></extra>")
    fig.add_vline(x=0, line_color="grey")
    fig.update_layout(title="SHAP beeswarm: how each feature pushes credibility, review by review",
                      xaxis_title="SHAP value (→ genuine, ← fake)",
                      yaxis=dict(tickvals=list(range(len(order))), ticktext=order), height=450)
    return fig


def shap_waterfall(row_factors: list, base: float) -> go.Figure:
    """row_factors: [(feature, value, contribution), ...] from shap_row_factors."""
    factors = sorted(row_factors, key=lambda t: abs(t[2]))
    labels = [f"{f} = {v:.3g}" for f, v, _ in factors]
    contribs = [c for _, _, c in factors]
    final = base + sum(contribs)
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(factors) + ["total"],
        y=["average review"] + labels + ["this review"],
        x=[base] + contribs + [final],
        text=[f"{base:.2f}"] + [f"{c:+.3f}" for c in contribs] + [f"{final:.2f}"],
        textposition="outside",
        increasing=dict(marker_color=GENUINE), decreasing=dict(marker_color=FAKE),
        totals=dict(marker_color="#34495e"), connector=dict(line=dict(color="lightgrey")),
    ))
    fig.update_layout(title=f"SHAP waterfall: from the average ({base:.2f}) to this review ({final:.2f})",
                      xaxis_title="P(genuine)", xaxis_range=[0, 1.05], height=400, showlegend=False)
    return fig


def lime_bar(weights: list, title: str = "LIME: local linear weights") -> go.Figure:
    """weights: [(label, weight), ...]."""
    df = pd.DataFrame(weights, columns=["label", "weight"]).iloc[::-1]
    fig = go.Figure(go.Bar(x=df["weight"], y=df["label"], orientation="h",
                           marker_color=[GENUINE if w > 0 else FAKE for w in df["weight"]],
                           text=[f"{w:+.3f}" for w in df["weight"]], textposition="outside"))
    fig.add_vline(x=0, line_color="grey")
    fig.update_layout(title=title, xaxis_title="weight (→ genuine, ← fake)", height=360)
    return fig


def agreement_heatmap(agreement: dict, metric: str = "top1_match") -> go.Figure:
    methods = ["zscore", "shap", "lime"]
    m = pd.DataFrame(1.0, index=methods, columns=methods)
    for key, v in agreement[metric].items():
        a, b = key.split("_vs_")
        m.loc[a, b] = m.loc[b, a] = v
    m.index = m.columns = [METHOD_LABELS[x] for x in methods]
    titles = {"top1_match": "Same #1 feature (share of reviews)", "top3_jaccard": "Top-3 overlap (Jaccard)",
              "kendall_tau": "Full ranking agreement (Kendall τ)"}
    fig = px.imshow(m, text_auto=".2f", color_continuous_scale="Blues", zmin=0, zmax=1,
                    title=titles.get(metric, metric))
    fig.update_layout(coloraxis_showscale=False, height=380)
    return fig


def faithfulness_curve(faithfulness: dict) -> go.Figure:
    fig = go.Figure()
    colors = {"shap": "#3498db", "lime": "#9b59b6", "zscore": "#e67e22", "random": "grey"}
    for method, values in faithfulness["prob_change"].items():
        fig.add_scatter(x=faithfulness["k"], y=values, mode="lines+markers", name=METHOD_LABELS[method],
                        line=dict(color=colors.get(method), dash="dash" if method == "random" else "solid"))
    fig.update_layout(title="Deletion test: prediction change when the top-k features are reset to the mean",
                      xaxis_title="k (features reset)", yaxis_title="mean |Δ P(genuine)|",
                      xaxis=dict(tickmode="linear", dtick=1), height=380)
    return fig


def class_contrast_bar(contrast: dict) -> go.Figure:
    rows = [{"feature": f, "class": c, "mean SHAP": v} for c, d in contrast.items() for f, v in d.items()]
    fig = px.bar(pd.DataFrame(rows), x="mean SHAP", y="feature", color="class", barmode="group",
                 orientation="h", color_discrete_map={"genuine": GENUINE, "fake": FAKE},
                 title="Average SHAP push per feature: genuine vs fake reviews")
    fig.add_vline(x=0, line_color="grey")
    fig.update_layout(yaxis_title=None, xaxis_title="mean SHAP value (→ genuine, ← fake)")
    return fig
