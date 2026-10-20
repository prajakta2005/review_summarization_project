"""
Publication / presentation figures in one minimal style, from results/*.json.

    python experiments/export_paper_figures.py

Writes docs/figures/paper/<name>.png (16:9, 2x scale, for slides) and
<name>.svg (vector, for the paper). Colours: blue = this system /
credibility-aware, orange = standard / baseline, grey = other.
Every chart says in text what the colours mean.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from results_summary import load_all_results

OUT_DIR = "docs/figures/paper"
INK, MUTED, GRID = "#1F2933", "#52606D", "#E4E7EB"
BLUE, ORANGE, GREY, LIGHT_BLUE = "#2E6FD8", "#E07A2E", "#9AA5B1", "#A9C4F0"
FONT = "Arial"
W, H = 1600, 900

FEATURE_LABELS = {"sentiment": "Sentiment", "length": "Length", "exclaim_density": "Exclamation density",
                  "repetition_ratio": "Repetition ratio", "specificity": "Specificity",
                  "rating_consistency": "Rating–text consistency"}
SET_LABELS = {"handcrafted": "6 credibility features", "tfidf": "TF-IDF text", "hybrid": "TF-IDF + features",
              "bert": "DistilBERT"}


def style(fig: go.Figure, title: str, subtitle: str = None, height: int = H) -> go.Figure:
    text = f"<b>{title}</b>" + (f"<br><span style='font-size:20px;color:{MUTED}'>{subtitle}</span>" if subtitle else "")
    fig.update_layout(
        title=dict(text=text, x=0.04, xanchor="left", y=0.95, font=dict(size=30, color=INK)),
        font=dict(family=FONT, size=20, color=INK),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        width=W, height=height, margin=dict(l=110, r=60, t=150, b=90),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=20)),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="", zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    return fig


def contamination_by_length(r):
    avg = r["summary_comparison"]["data"]["averages"]
    ns = sorted(avg, key=int)
    fig = go.Figure()
    for variant, name, color in [("standard", "Standard summary", ORANGE),
                                 ("credibility_aware", "Credibility-aware (ours)", BLUE)]:
        vals = [avg[n][variant]["contamination"] for n in ns]
        fig.add_bar(x=[f"{n} sentences" for n in ns], y=vals, name=name, marker_color=color,
                    text=[f"{v:.0%}" for v in vals], textposition="outside", textfont=dict(size=24))
    fig.update_layout(barmode="group", bargap=0.35, yaxis=dict(tickformat=".0%", range=[0, 1.1],
                                                               title="Summary sentences from fake reviews"))
    return style(fig, "Credibility weighting keeps most fake content out of summaries",
                 "Share of summary sentences quoted from ground-truth fake reviews · 10 categories · 40,412 reviews")


def contamination_per_category(r):
    rows = pd.DataFrame(r["summary_comparison"]["data"]["per_category"])
    rows = rows[rows["top_n"] == 5].pivot(index="category", columns="variant", values="contamination")
    rows = rows.sort_values("credibility_aware", ascending=False)
    labels = [c.replace("_5", "").replace("_", " ") for c in rows.index]
    fig = go.Figure()
    for i, label in enumerate(labels):
        fig.add_scatter(x=[rows["standard"].iloc[i], rows["credibility_aware"].iloc[i]], y=[label, label],
                        mode="lines", line=dict(color=GRID, width=6), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=rows["standard"], y=labels, mode="markers", name="Standard summary",
                    marker=dict(color=ORANGE, size=22))
    fig.add_scatter(x=rows["credibility_aware"], y=labels, mode="markers", name="Credibility-aware (ours)",
                    marker=dict(color=BLUE, size=22))
    fig.update_xaxes(tickformat=".0%", range=[-0.05, 1.08], showgrid=True, gridcolor=GRID,
                     title="Summary sentences from fake reviews (5-sentence summaries)")
    dropped = int((rows["credibility_aware"] < rows["standard"]).sum())
    return style(fig, f"Contamination drops in {dropped} of {len(rows)} categories",
                 "Each line joins the standard (orange) and credibility-aware (blue) summary of one category")


def rouge(r):
    avg = r["summary_comparison"]["data"]["averages"]["5"]
    metrics = [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")]
    fig = go.Figure()
    for variant, name, color in [("standard", "Standard summary", ORANGE),
                                 ("credibility_aware", "Credibility-aware (ours)", BLUE)]:
        vals = [avg[variant][m] for m, _ in metrics]
        fig.add_bar(x=[l for _, l in metrics], y=vals, name=name, marker_color=color,
                    text=[f"{v:.3f}" for v in vals], textposition="outside", textfont=dict(size=24))
    fig.update_layout(barmode="group", bargap=0.35, yaxis=dict(range=[0, 0.7], title="F-measure"))
    return style(fig, "Summaries move closer to what genuine reviewers say",
                 "ROUGE overlap with a genuine-only reference summary (higher is better) · 5-sentence summaries")


def threshold_tradeoff(r):
    d = r["summary_comparison"]["data"]
    avg5 = d["averages"]["5"]["credibility_aware"]
    rows = [{"setting": "Soft weighting<br>(ours, default)", **avg5}] + [
        {"setting": f"Drop reviews<br>below {s['hard_threshold']}", **s} for s in d["sensitivity"]["hard_threshold"]]
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_bar(x=df["setting"], y=df["contamination"], name="Fake contamination (lower is better)",
                marker_color=ORANGE, text=[f"{v:.0%}" for v in df["contamination"]], textposition="outside")
    fig.add_bar(x=df["setting"], y=df["rougeL"], name="ROUGE-L vs genuine-only (higher is better)",
                marker_color=BLUE, text=[f"{v:.3f}" for v in df["rougeL"]], textposition="outside")
    fig.update_layout(barmode="group", bargap=0.3, yaxis=dict(range=[0, 0.6]))
    return style(fig, "Hard filtering trades summary quality for purity",
                 "Credibility-aware variants, 5-sentence summaries, averaged over 10 categories")


def model_comparison_cv(r):
    cv = r["cv_results"]["data"]["models"]
    rows = [{"label": f"{v['model']} · {SET_LABELS[v['feature_set']]}", "feature_set": v["feature_set"],
             "mean": float(np.mean(v["f1"])), "std": float(np.std(v["f1"]))} for v in cv.values()]
    df = pd.DataFrame(rows).sort_values("mean")
    colors = [BLUE if fs == "handcrafted" else GREY for fs in df["feature_set"]]
    fig = go.Figure(go.Bar(x=df["mean"], y=df["label"], orientation="h", marker_color=colors,
                           error_x=dict(type="data", array=df["std"], color=INK, thickness=2),
                           text=[f"{m:.3f}" for m in df["mean"]], textposition="outside", textfont=dict(size=20)))
    fig.update_xaxes(range=[0.6, 1.02], title="F1 (5-fold cross-validation, mean ± std)", showgrid=True,
                     gridcolor=GRID)
    fig.update_layout(margin=dict(l=470))
    best = df["mean"].max()
    hand = df[df["feature_set"] == "handcrafted"]["mean"]
    gap_lo, gap_hi = best - hand.max(), best - hand.min()
    return style(fig, f"Text models classify best; explainable models trail by {gap_lo:.2f}–{gap_hi:.2f} F1",
                 "Blue = interpretable 6-feature models · grey = text models · 40,412 reviews")


SHORT = {"rf_handcrafted": ("RF (deployed)", "bottom center"), "rf_handcrafted_tuned": ("RF tuned", "top center"),
         "lr_handcrafted": ("LR", "top center"), "svm_handcrafted": ("SVM", "top center"),
         "rf_tfidf": ("RF · text", "bottom center"), "lr_tfidf": ("LR · text", "bottom left"),
         "svm_tfidf": ("SVM · text", "top left"), "lr_hybrid": ("LR · text+features", "bottom right"),
         "svm_hybrid": ("SVM · text+features", "top center"), "distilbert": ("DistilBERT", "top center")}


def tradeoff_scatter(r):
    models = r["model_comparison"]["data"]["models"]
    fig = go.Figure()
    for interp, name, color in [(True, "Interpretable (6 credibility features)", BLUE),
                                (False, "Text-based (TF-IDF)", GREY)]:
        sub = [m for m in models if m["interpretable"] == interp]
        fig.add_scatter(x=[m["train_seconds"] for m in sub], y=[m["f1"] for m in sub], mode="markers+text",
                        name=name, marker=dict(size=24, color=color),
                        text=[SHORT.get(m["model_id"], (m["model"], ""))[0] for m in sub],
                        textposition=[SHORT.get(m["model_id"], ("", "top center"))[1] for m in sub],
                        textfont=dict(size=20))
    ticks = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    fig.update_xaxes(type="log", title="Training time (seconds, log scale)", showgrid=True, gridcolor=GRID,
                     tickvals=ticks, ticktext=[str(t) for t in ticks], minor=dict(showgrid=False, ticks=""))
    fig.update_yaxes(range=[0.65, 1.0], title="F1 on the shared test set")
    return style(fig, "Accuracy vs explainability vs cost",
                 "The deployed model is the interpretable Random Forest: every score can be explained in 6 signals")


def confusion_pair(r):
    d = r["model_comparison"]["data"]
    by_id = {m["model_id"]: m for m in d["models"]}
    pairs = [(by_id[d["deployed_model"]], "Deployed: Random Forest · 6 features"),
             (by_id[d["best_by_f1"]], "Best: linear SVM · TF-IDF + features")]
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=[p[1] for p in pairs], horizontal_spacing=0.12)
    for i, (m, _) in enumerate(pairs, start=1):
        cm = np.array(m["confusion_matrix"], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        text = [[f"{norm[a, b]:.0%}<br>{int(cm[a, b]):,}" for b in range(2)] for a in range(2)]
        fig.add_heatmap(z=norm, x=["predicted fake", "predicted genuine"], y=["actual fake", "actual genuine"],
                        text=text, texttemplate="%{text}", textfont=dict(size=26),
                        colorscale=[[0, "#F5F7FA"], [1, BLUE]], zmin=0, zmax=1, showscale=False, row=1, col=i)
        fig.update_yaxes(autorange="reversed", row=1, col=i)
    fig.update_annotations(font=dict(size=22, color=INK))
    return style(fig, "Where the models go wrong",
                 "Row-normalised confusion matrices on the same 8,083 held-out reviews")


def ablation(r):
    rf = r["ablation"]["data"]["models"]["rf"]
    df = pd.DataFrame([{"feature": FEATURE_LABELS[x["removed"]], "delta": x["delta_f1"],
                        "std": x["f1"]["std"]} for x in rf["leave_one_out"]]).sort_values("delta", ascending=False)
    colors = [ORANGE if d < -0.01 else GREY for d in df["delta"]]
    fig = go.Figure(go.Bar(x=df["delta"], y=df["feature"], orientation="h", marker_color=colors,
                           error_x=dict(type="data", array=df["std"], color=INK, thickness=2),
                           text=[f"{d:+.3f}" for d in df["delta"]], textposition="outside", textfont=dict(size=22)))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_xaxes(range=[-0.25, 0.05], title="Change in F1 when the feature is removed", showgrid=True,
                     gridcolor=GRID)
    fig.update_layout(margin=dict(l=330))
    return style(fig, "Two signals carry the model: repetition and length",
                 f"Leave-one-out ablation, Random Forest, 5-fold CV (full model F1 = {rf['full']['f1']['mean']:.3f})")


def faithfulness(r):
    f = r["explainability_eval"]["data"]["faithfulness"]
    names = {"shap": ("SHAP", BLUE, "solid"), "lime": ("LIME", LIGHT_BLUE, "solid"),
             "zscore": ("z-score (original)", ORANGE, "solid"), "random": ("Random feature (baseline)", GREY, "dash")}
    fig = go.Figure()
    for key, (name, color, dash) in names.items():
        fig.add_scatter(x=f["k"], y=f["prob_change"][key], name=name, mode="lines+markers",
                        line=dict(color=color, width=5, dash=dash), marker=dict(size=16))
    fig.update_xaxes(tickvals=f["k"], title="Number of top-ranked features reset to the dataset mean")
    fig.update_yaxes(range=[0, 0.16], title="Mean change in predicted credibility")
    return style(fig, "SHAP explanations are the most faithful to the model",
                 "Deletion test on 500 held-out reviews: a faithful explanation's top features move the prediction most")


def agreement(r):
    ag = r["explainability_eval"]["data"]["agreement"]["top1_match"]
    methods = ["z-score", "SHAP", "LIME"]
    key = {"z-score": "zscore", "SHAP": "shap", "LIME": "lime"}
    z = np.ones((3, 3))
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if i != j:
                k1, k2 = f"{key[a]}_vs_{key[b]}", f"{key[b]}_vs_{key[a]}"
                z[i, j] = ag.get(k1, ag.get(k2))
    fig = go.Figure(go.Heatmap(z=z, x=methods, y=methods, text=[[f"{v:.0%}" for v in row] for row in z],
                               texttemplate="%{text}", textfont=dict(size=30),
                               colorscale=[[0, "#F5F7FA"], [1, BLUE]], zmin=0, zmax=1, showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(width=1100, margin=dict(l=160))
    fig = style(fig, "The original explanations rarely match the model",
                "Share of reviews where two methods pick the same most important feature")
    fig.update_layout(width=1100)
    return fig


def importance_rankings(r):
    vals = r["ablation"]["data"]["rankings"]["values"]
    methods = [("ablation_loo", "Ablation (F1 drop)", ORANGE), ("shap", "Mean |SHAP|", BLUE),
               ("permutation", "Permutation", LIGHT_BLUE), ("impurity", "RF impurity", GREY)]
    order = sorted(vals["ablation_loo"], key=lambda f: -vals["ablation_loo"][f])
    fig = go.Figure()
    for key, name, color in methods:
        if key not in vals:
            continue
        v = pd.Series(vals[key]).reindex(order)
        v = v / v.abs().max()
        fig.add_bar(x=[FEATURE_LABELS[f] for f in order], y=v, name=name, marker_color=color)
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(title="Importance (scaled to the top feature = 1)")
    return style(fig, "Every importance measure ranks repetition first",
                 "Four ways of measuring feature importance, each scaled to its own maximum")


def feature_separation(r):
    stats = r["dataset_profile"]["data"]["feature_stats_by_label"]
    df = pd.DataFrame([{"feature": FEATURE_LABELS[f], "d": s["cohens_d"]} for f, s in stats.items()]).sort_values("d")
    colors = [BLUE if d > 0 else ORANGE for d in df["d"]]
    fig = go.Figure(go.Bar(x=df["d"], y=df["feature"], orientation="h", marker_color=colors,
                           text=[f"{d:+.2f}" for d in df["d"]], textposition="outside", textfont=dict(size=22)))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_xaxes(range=[-0.65, 0.35], title="Cohen's d (genuine minus fake)", showgrid=True, gridcolor=GRID)
    fig.update_layout(margin=dict(l=330))
    return style(fig, "Fake reviews repeat themselves",
                 "Blue = higher in genuine reviews · orange = higher in fake reviews · 40,412 reviews")


def capability_matrix(r):
    caps = ["Detects fake<br>reviews", "Explains each<br>decision", "Summarizes<br>reviews",
            "Credibility-<br>weighted summary", "Sentence → review<br>→ reasons trace", "Measures fake<br>content in summary"]
    rows = [("Fake-review detection", [1, 0, 0, 0, 0, 0]),
            ("Review summarization", [0, 0, 1, 0, 0, 0]),
            ("Explainable ML (SHAP, LIME)", [0, 1, 0, 0, 0, 0]),
            ("This system", [1, 1, 1, 1, 1, 1])]
    z = np.array([v for _, v in rows], dtype=float)
    z[-1] = z[-1] * 2  # highlight our row
    text = [["Yes" if c else "No" for c in v] for _, v in rows]
    fig = go.Figure(go.Heatmap(z=z, x=caps, y=[n for n, _ in rows], text=text, texttemplate="%{text}",
                               textfont=dict(size=24), xgap=6, ygap=6, showscale=False,
                               colorscale=[[0, "#F5F7FA"], [0.5, LIGHT_BLUE], [1, BLUE]], zmin=0, zmax=2))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top")
    fig.update_layout(margin=dict(l=380, t=230))
    return style(fig, "Capability comparison", "Representative families of existing approaches vs this system")


FIGURES = [
    ("P01_contamination_by_length", contamination_by_length),
    ("P02_contamination_per_category", contamination_per_category),
    ("P03_rouge", rouge),
    ("P04_threshold_tradeoff", threshold_tradeoff),
    ("P05_model_comparison_cv", model_comparison_cv),
    ("P06_accuracy_vs_explainability", tradeoff_scatter),
    ("P07_confusion_matrices", confusion_pair),
    ("P08_ablation", ablation),
    ("P09_explanation_faithfulness", faithfulness),
    ("P10_explanation_agreement", agreement),
    ("P11_importance_rankings", importance_rankings),
    ("P12_feature_separation", feature_separation),
    ("P13_capability_matrix", capability_matrix),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = load_all_results()
    for name, build in FIGURES:
        fig = build(results)
        for ext in ("png", "svg"):
            # kaleido>=1.0 ignores layout width/height, so pass the size explicitly
            fig.write_image(os.path.join(OUT_DIR, f"{name}.{ext}"), width=fig.layout.width or W,
                            height=fig.layout.height or H, scale=2 if ext == "png" else 1)
        print(f"wrote {OUT_DIR}/{name}.png/.svg")


if __name__ == "__main__":
    main()
