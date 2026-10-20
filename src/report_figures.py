"""
report_figures.py
-----------------
The report's figures (F1-F10), built from results/*.json with the same
chart builders the dashboard pages use. Shared by the Experimental
Results page and experiments/export_figures.py.
"""

from charts import dataset, model_comparison, performance, ablation, explainability, summarization
from features import FEATURE_COLUMNS


def report_figures(results: dict) -> dict:
    """{file_stem: (caption, plotly Figure)} for every figure whose results exist."""
    figs = {}

    def data(name):
        r = results.get(name)
        return r["data"] if r else None

    profile = data("dataset_profile")
    if profile:
        figs["F1a_class_balance"] = ("Class balance of the full dataset", dataset.class_balance_bar(profile))
        figs["F1b_categories"] = ("Genuine vs fake reviews per category", dataset.category_stacked_bar(profile))

    comparison = data("model_comparison")
    if comparison:
        models = comparison["models"]
        by_id = {m["model_id"]: m for m in models}
        figs["F2_leaderboard"] = ("Model leaderboard by F1 (shared test set)",
                                  model_comparison.leaderboard_bar(models, "f1", comparison["deployed_model"]))
        figs["F3_cost_vs_quality"] = ("F1 vs training time", model_comparison.cost_vs_quality_scatter(models))
        deployed = by_id[comparison["deployed_model"]]
        best = by_id[comparison["best_by_f1"]]
        figs["F4a_confusion_deployed"] = (
            "Confusion matrix, deployed Random Forest",
            performance.confusion_matrix_heatmap(deployed["confusion_matrix"], True,
                                                 title="Deployed Random Forest (6 features)"))
        figs["F4b_confusion_best"] = (
            f"Confusion matrix, best model ({best['model']} · {best['feature_set']})",
            performance.confusion_matrix_heatmap(best["confusion_matrix"], True,
                                                 title=f"{best['model']} · {best['feature_set']}"))
        figs["F5_roc_curves"] = ("ROC curves of all models", performance.roc_curves(models))

    abl = data("ablation")
    if abl:
        rf = abl["models"]["rf"]
        figs["F6_ablation"] = ("Change in F1 when each feature is removed (Random Forest, 5-fold CV)",
                               ablation.delta_bar(rf["leave_one_out"], rf["full"]))

    explain = data("explainability_eval")
    if explain:
        figs["F7_global_shap"] = ("Global SHAP importance", explainability.shap_bar(explain["global_shap"]))
        fake = next((e for e in explain["examples"] if e["label"] == "fake"), None)
        if fake:
            factors = [(f, None, fake["shap"][f]) for f in FEATURE_COLUMNS]
            fig = explainability.shap_waterfall(factors, explain["base_value"])
            figs["F8_shap_waterfall_fake"] = (f"SHAP waterfall for a fake review (#{fake['review_id']})", fig)

    summary = data("summary_comparison")
    if summary:
        figs["F9_contamination"] = ("Fake contamination per category, standard vs credibility-aware",
                                    summarization.contamination_bar(summary["per_category"], 5))
        avg = summary["averages"].get("5") or next(iter(summary["averages"].values()))
        figs["F10_rouge"] = ("ROUGE against the genuine-only reference", summarization.rouge_grouped_bar(avg))
    return figs
