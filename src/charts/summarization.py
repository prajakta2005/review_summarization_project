"""
Summarization comparison charts. Builders take data from
results/summary_comparison.json (or live evaluate_summaries output).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

VARIANT_COLORS = {"standard": "#95a5a6", "credibility_aware": "#2ecc71", "oracle": "#3498db"}
VARIANT_LABELS = {"standard": "Standard", "credibility_aware": "Credibility-aware (ours)",
                  "oracle": "Genuine-only (oracle)"}
METRIC_LABELS = {"contamination": "Fake contamination", "mean_source_cred": "Mean source credibility",
                 "rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L",
                 "redundancy": "Redundancy", "coverage": "Attribute coverage"}


def _label(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(summary=df["variant"].map(VARIANT_LABELS))


def contamination_bar(per_category: list, top_n: int = 5) -> go.Figure:
    df = _label(pd.DataFrame(per_category))
    df = df[df["top_n"] == top_n]
    fig = px.bar(df, x="category", y="contamination", color="summary", barmode="group",
                 color_discrete_map={VARIANT_LABELS[k]: v for k, v in VARIANT_COLORS.items()},
                 title=f"Share of summary sentences taken from FAKE reviews ({top_n}-sentence summaries)")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="fake contamination (lower is better)",
                      xaxis_title=None, legend_title=None)
    return fig


def rouge_grouped_bar(averages: dict) -> go.Figure:
    """averages: {variant: {metric: value}} for one top_n."""
    rows = [{"summary": VARIANT_LABELS[v], "metric": METRIC_LABELS[m], "value": vals[m]}
            for v, vals in averages.items() for m in ["rouge1", "rouge2", "rougeL"]]
    fig = px.bar(pd.DataFrame(rows), x="metric", y="value", color="summary", barmode="group", text="value",
                 color_discrete_map={VARIANT_LABELS[k]: v for k, v in VARIANT_COLORS.items()},
                 title="Overlap with the genuine-only reference summary")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis_title="F-measure (higher is better)", xaxis_title=None, legend_title=None)
    return fig


def metric_delta_table(averages: dict) -> pd.DataFrame:
    std, ours = averages["standard"], averages["credibility_aware"]
    better_low = {"contamination", "redundancy"}
    rows = []
    for m, label in METRIC_LABELS.items():
        if std.get(m) is None or ours.get(m) is None:
            continue
        delta = ours[m] - std[m]
        improved = delta < 0 if m in better_low else delta > 0
        rows.append({"metric": label, "standard": round(std[m], 4), "credibility-aware": round(ours[m], 4),
                     "change": round(delta, 4), "better?": "✅" if improved else ("—" if abs(delta) < 1e-4 else "❌")})
    return pd.DataFrame(rows)


def sensitivity_line(rows: list, param: str) -> go.Figure:
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_scatter(x=df[param], y=df["contamination"], mode="lines+markers", name="fake contamination",
                    line=dict(color="#e74c3c"))
    fig.add_scatter(x=df[param], y=df["rougeL"], mode="lines+markers", name="ROUGE-L", line=dict(color="#3498db"))
    titles = {"weight_floor": "Effect of the minimum sentence weight",
              "hard_threshold": "Effect of dropping reviews below a credibility score"}
    fig.update_layout(title=titles.get(param, param), xaxis_title=param.replace("_", " "),
                      yaxis_title="value", yaxis_range=[0, 1])
    return fig


def topn_line(averages_by_topn: dict, metric: str = "contamination") -> go.Figure:
    rows = [{"top_n": int(n), "summary": VARIANT_LABELS[v], metric: vals[metric]}
            for n, by_variant in averages_by_topn.items() for v, vals in by_variant.items()]
    fig = px.line(pd.DataFrame(rows), x="top_n", y=metric, color="summary", markers=True,
                  color_discrete_map={VARIANT_LABELS[k]: v for k, v in VARIANT_COLORS.items()},
                  title=f"{METRIC_LABELS[metric]} vs summary length")
    fig.update_layout(xaxis=dict(tickmode="linear", dtick=1), xaxis_title="sentences in summary",
                      legend_title=None)
    if metric == "contamination":
        fig.update_layout(yaxis_tickformat=".0%")
    return fig
