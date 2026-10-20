"""
Dataset charts. Every builder takes the dict from
dataset_profile.profile_dataset() (live) or results/dataset_profile.json
(saved), so the same chart works for both.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

LABEL_COLORS = {"genuine": "#2ecc71", "fake": "#e74c3c", "unlabeled": "#95a5a6"}


def _groups(profile: dict) -> list:
    return list(profile["label_counts"].keys())


def class_balance_bar(profile: dict) -> go.Figure:
    counts = pd.DataFrame(
        {"label": list(profile["label_counts"]), "reviews": list(profile["label_counts"].values())}
    )
    fig = px.bar(counts, x="label", y="reviews", color="label", text="reviews",
                 color_discrete_map=LABEL_COLORS, title="Class balance")
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="Reviews", xaxis_title=None)
    return fig


def category_donut(profile: dict) -> go.Figure:
    cats = pd.DataFrame(profile["categories"])
    fig = px.pie(cats, names="category", values="total", hole=0.5, title="Reviews per category")
    fig.update_traces(textinfo="percent")
    return fig


def category_stacked_bar(profile: dict) -> go.Figure:
    cats = pd.DataFrame(profile["categories"]).sort_values("total")
    long = cats.melt(id_vars="category", value_vars=_groups(profile),
                     var_name="label", value_name="reviews")
    fig = px.bar(long, y="category", x="reviews", color="label", orientation="h",
                 color_discrete_map=LABEL_COLORS, title="Genuine vs fake reviews per category")
    fig.update_layout(yaxis_title=None, barmode="stack")
    return fig


def rating_by_label_bar(profile: dict) -> go.Figure:
    rows = [{"rating": int(r), "label": g, "reviews": c}
            for g, ratings in profile["rating_by_label"].items() for r, c in ratings.items()]
    fig = px.bar(pd.DataFrame(rows), x="rating", y="reviews", color="label", barmode="group",
                 color_discrete_map=LABEL_COLORS, title="Star rating by label")
    fig.update_layout(xaxis=dict(tickmode="linear", dtick=1))
    return fig


def feature_hist(profile: dict, feature: str, normalize: bool = True) -> go.Figure:
    """Overlaid pre-binned histograms of one feature per label. With
    normalize=True each label's bars sum to 1, so shapes are comparable."""
    hist = profile["feature_hist_by_label"][feature]
    edges = hist["bins"]
    centers = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    width = edges[1] - edges[0] if len(edges) > 1 else 1
    fig = go.Figure()
    for g in _groups(profile):
        counts = hist[g]
        total = sum(counts) or 1
        y = [c / total for c in counts] if normalize else counts
        fig.add_bar(x=centers, y=y, width=width, name=g, opacity=0.6,
                    marker_color=LABEL_COLORS.get(g))
    fig.update_layout(barmode="overlay", title=f"Distribution of {feature} by label",
                      xaxis_title=f"{feature} (top 1% clipped)",
                      yaxis_title="share of reviews" if normalize else "reviews")
    return fig


def feature_violin(df: pd.DataFrame, feature: str, label_col: str = "label_name") -> go.Figure:
    """Violin of one feature per label from (a sample of) the live dataframe."""
    fig = px.violin(df, x=label_col, y=feature, color=label_col, box=True,
                    color_discrete_map=LABEL_COLORS, title=f"{feature}: genuine vs fake")
    fig.update_layout(showlegend=False, xaxis_title=None)
    return fig


def separation_bar(profile: dict) -> go.Figure:
    """Cohen's d per feature (genuine minus fake): a quick preview of which
    signals separate the classes. The ablation study confirms it properly."""
    rows = [{"feature": f, "cohens_d": s["cohens_d"]}
            for f, s in profile["feature_stats_by_label"].items() if "cohens_d" in s]
    d = pd.DataFrame(rows).sort_values("cohens_d", key=abs)
    d["direction"] = d["cohens_d"].apply(lambda v: "higher in genuine" if v > 0 else "higher in fake")
    fig = px.bar(d, x="cohens_d", y="feature", orientation="h", color="direction",
                 color_discrete_map={"higher in genuine": LABEL_COLORS["genuine"],
                                     "higher in fake": LABEL_COLORS["fake"]},
                 title="How well each feature separates genuine from fake (Cohen's d)")
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    fig.update_layout(yaxis_title=None)
    return fig
