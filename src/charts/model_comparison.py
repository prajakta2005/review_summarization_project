"""
Model comparison charts. Every builder takes the `models` list from
results/model_comparison.json (one dict per model).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

FEATURE_SET_COLORS = {"handcrafted": "#3498db", "tfidf": "#e67e22", "hybrid": "#9b59b6", "bert": "#e74c3c"}
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                 "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1_macro": "Macro F1"}


def to_frame(models: list) -> pd.DataFrame:
    df = pd.DataFrame(models)
    df["label"] = df["model"] + " · " + df["feature_set"]
    return df


def leaderboard_bar(models: list, metric: str = "f1", deployed: str = "rf_handcrafted") -> go.Figure:
    df = to_frame(models).sort_values(metric)
    df["is_deployed"] = df["model_id"].eq(deployed).map({True: "  ◀ deployed", False: ""})
    fig = px.bar(df, x=metric, y="label", color="feature_set", orientation="h",
                 color_discrete_map=FEATURE_SET_COLORS, text=metric,
                 title=f"Leaderboard by {METRIC_LABELS.get(metric, metric)} (same test set for every model)")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
    lo = max(0.0, df[metric].min() - 0.1)
    fig.update_layout(xaxis_range=[lo, 1.02], yaxis_title=None,
                      xaxis_title=METRIC_LABELS.get(metric, metric), legend_title="Feature set",
                      height=120 + 40 * len(df))
    for _, row in df[df["is_deployed"] != ""].iterrows():
        fig.add_annotation(x=lo, y=row["label"], text="deployed ▶", showarrow=False,
                           xanchor="right", font=dict(size=11, color="grey"))
    return fig


def metric_heatmap(models: list) -> go.Figure:
    df = to_frame(models).sort_values("f1", ascending=False).set_index("label")[METRICS]
    fig = px.imshow(df, text_auto=".3f", aspect="auto", color_continuous_scale="Blues",
                    labels=dict(color="score"), title="All metrics per model")
    fig.update_xaxes(ticktext=[METRIC_LABELS[m] for m in METRICS], tickvals=list(range(len(METRICS))))
    fig.update_layout(yaxis_title=None, xaxis_title=None, height=120 + 40 * len(df))
    return fig


def cost_vs_quality_scatter(models: list, metric: str = "f1") -> go.Figure:
    """Accuracy-vs-cost-vs-explainability trade-off in one view."""
    df = to_frame(models).dropna(subset=["train_seconds"])
    df["explainable"] = df["interpretable"].map({True: "interpretable features", False: "text / black-box"})
    fig = px.scatter(df, x="train_seconds", y=metric, color="feature_set", symbol="explainable",
                     hover_name="label", log_x=True, size=[14] * len(df),
                     color_discrete_map=FEATURE_SET_COLORS,
                     hover_data={"predict_seconds": True, "train_seconds": ":.1f"},
                     title=f"{METRIC_LABELS.get(metric, metric)} vs training time")
    fig.update_layout(xaxis_title="training time (s, log scale)", yaxis_title=METRIC_LABELS.get(metric, metric))
    return fig


def featureset_grouped_bar(models: list, metric: str = "f1") -> go.Figure:
    """Model family × feature set: does the representation matter more than the classifier?"""
    df = to_frame(models)
    df = df[df["feature_set"] != "bert"]
    df["family"] = df["model"].str.replace(r" \(.*\)", "", regex=True)
    df = df.sort_values(metric, ascending=False).drop_duplicates(["family", "feature_set"])
    fig = px.bar(df, x="family", y=metric, color="feature_set", barmode="group", text=metric,
                 color_discrete_map=FEATURE_SET_COLORS,
                 category_orders={"feature_set": ["handcrafted", "tfidf", "hybrid"]},
                 title=f"{METRIC_LABELS.get(metric, metric)} by model family and feature set")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis_range=[max(0.0, df[metric].min() - 0.1), 1.02], xaxis_title=None)
    return fig
