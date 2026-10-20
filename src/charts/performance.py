"""
Performance charts: confusion matrix, ROC / PR curves, threshold sweep,
per-category F1 and cross-validation spread.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from charts.model_comparison import FEATURE_SET_COLORS

CLASS_LABELS = ["fake", "genuine"]


def confusion_matrix_heatmap(cm, normalize: bool = False, title: str = "Confusion matrix") -> go.Figure:
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    text = [[f"{shown[i, j]:.1%}<br>({int(cm[i, j]):,})" if normalize else f"{int(cm[i, j]):,}"
             for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=shown, x=[f"predicted {c}" for c in CLASS_LABELS], y=[f"actual {c}" for c in CLASS_LABELS],
        text=text, texttemplate="%{text}", textfont=dict(size=16),
        colorscale="Blues", showscale=False,
    ))
    fig.update_layout(title=title, yaxis_autorange="reversed", height=380)
    return fig


def _curve_figure(models: list, key: str, x: str, y: str, title: str, auc_key: str) -> go.Figure:
    fig = go.Figure()
    for m in models:
        curve = m.get(key)
        if not curve:
            continue
        fig.add_scatter(x=curve[x], y=curve[y], mode="lines",
                        name=f"{m['model']} · {m['feature_set']} ({m[auc_key]:.3f})",
                        line=dict(color=FEATURE_SET_COLORS.get(m["feature_set"]),
                                  dash="solid" if m["model"].startswith(("Logistic", "DistilBERT")) else
                                  "dot" if m["model"].startswith("SVM") else "dash"))
    fig.update_layout(title=title, legend=dict(font=dict(size=10)), height=450)
    return fig


def roc_curves(models: list) -> go.Figure:
    fig = _curve_figure(models, "roc_curve", "fpr", "tpr", "ROC curves (AUC in legend)", "roc_auc")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="grey", dash="dash"),
                    name="chance", showlegend=False)
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    return fig


def pr_curves(models: list) -> go.Figure:
    fig = _curve_figure(models, "pr_curve", "recall", "precision",
                        "Precision–recall curves (PR-AUC in legend)", "pr_auc")
    fig.update_layout(xaxis_title="Recall (genuine)", yaxis_title="Precision (genuine)")
    return fig


def threshold_curve(sweep: pd.DataFrame, current: float = None, markers: dict = None) -> go.Figure:
    """markers: {label: threshold} drawn as vertical lines (e.g. Low/High cut-offs)."""
    fig = go.Figure()
    for metric, color in [("precision", "#3498db"), ("recall", "#e67e22"),
                          ("f1", "#2ecc71"), ("accuracy", "#7f8c8d")]:
        fig.add_scatter(x=sweep["threshold"], y=sweep[metric], mode="lines", name=metric,
                        line=dict(color=color))
    for label, t in (markers or {}).items():
        fig.add_vline(x=t, line_dash="dot", line_color="grey", annotation_text=label,
                      annotation_position="top")
    if current is not None:
        fig.add_vline(x=current, line_color="black", annotation_text="selected",
                      annotation_position="bottom right")
    fig.update_layout(title="Metrics vs decision threshold", xaxis_title="threshold on genuine score",
                      yaxis_title="score", yaxis_range=[0, 1.02], height=420)
    return fig


def per_category_bar(cat_df: pd.DataFrame, metric: str = "f1") -> go.Figure:
    fig = px.bar(cat_df, x=metric, y="category", orientation="h", text=metric,
                 color=metric, color_continuous_scale="RdYlGn", hover_data=["n", "accuracy"],
                 title=f"{metric.upper()} per category")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_title=None, coloraxis_showscale=False,
                      xaxis_range=[max(0.0, cat_df[metric].min() - 0.1), 1.02])
    return fig


def cv_box(cv_models: dict, metric: str = "f1") -> go.Figure:
    rows = [{"model": f"{v['model']} · {v['feature_set']}", "feature_set": v["feature_set"], metric: s}
            for v in cv_models.values() for s in v[metric]]
    df = pd.DataFrame(rows)
    order = df.groupby("model")[metric].mean().sort_values().index.tolist()
    fig = px.box(df, x=metric, y="model", color="feature_set", points="all",
                 color_discrete_map=FEATURE_SET_COLORS, category_orders={"model": order},
                 title=f"{metric.upper()} across cross-validation folds")
    fig.update_layout(yaxis_title=None, height=120 + 45 * len(order))
    return fig
