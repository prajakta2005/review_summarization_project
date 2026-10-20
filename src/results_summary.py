"""
results_summary.py
------------------
Pulls every experiment's results/*.json into headline numbers and the
report's master tables (T1-T7). The Experimental Results page and
experiments/export_tables.py both use this, so the dashboard and the
report always show the same numbers.
"""

import pandas as pd

import results_io

RESULT_NAMES = ["dataset_profile", "model_comparison", "cv_results", "significance",
                "ablation", "explainability_eval", "summary_comparison"]


def load_all_results() -> dict:
    return {name: results_io.load_result(name) for name in RESULT_NAMES}


def _data(results: dict, name: str):
    r = results.get(name)
    return r["data"] if r else None


def headline_numbers(results: dict = None) -> dict:
    """Key numbers for the report; missing experiments give None."""
    results = results if results is not None else load_all_results()
    out = {}

    profile = _data(results, "dataset_profile")
    if profile:
        out["n_reviews"] = profile["n_after_preprocess"]
        out["n_categories"] = profile["n_categories"]

    comparison = _data(results, "model_comparison")
    if comparison:
        by_id = {m["model_id"]: m for m in comparison["models"]}
        best = by_id[comparison["best_by_f1"]]
        deployed = by_id.get(comparison["deployed_model"])
        out["best_model"] = f"{best['model']} · {best['feature_set']}"
        out["best_f1"] = best["f1"]
        out["deployed_f1"] = deployed["f1"] if deployed else None
        out["bert_included"] = "distilbert" in by_id

    ablation = _data(results, "ablation")
    if ablation:
        loo = ablation["models"]["rf"]["leave_one_out"]
        out["top_feature"] = loo[0]["removed"]
        out["top_feature_delta_f1"] = loo[0]["delta_f1"]

    explain = _data(results, "explainability_eval")
    if explain:
        out["shap_zscore_agreement"] = explain["agreement"]["top1_match"]["zscore_vs_shap"]
        pc = explain["faithfulness"]["prob_change"]
        out["shap_faithfulness_k1"] = pc["shap"][0]
        out["random_faithfulness_k1"] = pc["random"][0]

    summary = _data(results, "summary_comparison")
    if summary:
        avg = summary["averages"].get("5") or next(iter(summary["averages"].values()))
        out["contamination_standard"] = avg["standard"]["contamination"]
        out["contamination_ours"] = avg["credibility_aware"]["contamination"]
        out["rougeL_standard"] = avg["standard"]["rougeL"]
        out["rougeL_ours"] = avg["credibility_aware"]["rougeL"]
    return out


def key_findings(h: dict) -> list:
    """Sentences generated from headline_numbers, so they never go stale."""
    findings = []
    if "contamination_standard" in h:
        findings.append(
            f"Credibility-aware summarization cut fake-sentence contamination from "
            f"{h['contamination_standard']:.0%} to {h['contamination_ours']:.0%} and raised ROUGE-L against a "
            f"genuine-only reference from {h['rougeL_standard']:.3f} to {h['rougeL_ours']:.3f} "
            f"(5-sentence summaries, averaged over categories).")
    if "best_f1" in h:
        findings.append(
            f"The best classifier is {h['best_model']} (F1 {h['best_f1']:.3f}); the deployed, explainable "
            f"Random Forest on 6 features reaches F1 {h['deployed_f1']:.3f}."
            + ("" if h.get("bert_included") else " DistilBERT results are still pending (Colab)."))
    if "top_feature" in h:
        findings.append(
            f"Ablation: removing {h['top_feature']} costs the most F1 ({h['top_feature_delta_f1']:+.3f}).")
    if "shap_zscore_agreement" in h:
        findings.append(
            f"SHAP explanations are faithful to the model (resetting the top feature moves the prediction by "
            f"{h['shap_faithfulness_k1']:.3f} vs {h['random_faithfulness_k1']:.3f} for a random feature); the "
            f"original z-score explanations pick the same top feature only {h['shap_zscore_agreement']:.0%} of the time.")
    if "n_reviews" in h:
        findings.append(f"All results use the full dataset: {h['n_reviews']:,} reviews across "
                        f"{h['n_categories']} categories.")
    return findings


def _mean_std(values) -> str:
    s = pd.Series(values)
    return f"{s.mean():.4f} ± {s.std(ddof=0):.4f}"


def master_tables(results: dict = None) -> dict:
    """{table_name: DataFrame} for every experiment that has results."""
    results = results if results is not None else load_all_results()
    tables = {}

    profile = _data(results, "dataset_profile")
    if profile:
        rows = [("Raw reviews", profile["n_raw"]), ("After preprocessing", profile["n_after_preprocess"]),
                ("Exact duplicates removed", profile["duplicates_removed"]),
                ("Categories", profile["n_categories"])]
        rows += [(f"{label.capitalize()} reviews", n) for label, n in profile["label_counts"].items()]
        tables["T1_dataset"] = pd.DataFrame(rows, columns=["statistic", "value"])

    comparison = _data(results, "model_comparison")
    if comparison:
        cols = ["model", "feature_set", "accuracy", "precision", "recall", "f1", "roc_auc", "train_seconds"]
        tables["T2_model_comparison"] = pd.DataFrame(comparison["models"])[cols]

    cv = _data(results, "cv_results")
    if cv:
        tables["T3_cross_validation"] = pd.DataFrame([
            {"model": v["model"], "feature_set": v["feature_set"],
             **{m: _mean_std(v[m]) for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}}
            for v in cv["models"].values()
        ])

    sig = _data(results, "significance")
    if sig:
        t = pd.DataFrame(sig["pairs"])
        tables["T4_significance"] = t[["model_a", "model_b", "accuracy_a", "accuracy_b", "statistic",
                                       "p_value", "significant"]].round(4)

    ablation = _data(results, "ablation")
    if ablation:
        rf = ablation["models"]["rf"]
        tables["T5_ablation"] = pd.DataFrame(
            [{"feature removed": "(none — all 6)", "f1": f"{rf['full']['f1']['mean']:.4f} ± {rf['full']['f1']['std']:.4f}",
              "delta_f1": 0.0}] +
            [{"feature removed": r["removed"], "f1": f"{r['f1']['mean']:.4f} ± {r['f1']['std']:.4f}",
              "delta_f1": r["delta_f1"]} for r in rf["leave_one_out"]])

    explain = _data(results, "explainability_eval")
    if explain:
        ag = explain["agreement"]
        tables["T6_explanation_agreement"] = pd.DataFrame([
            {"methods": pair.replace("_vs_", " vs "), "top-1 match": ag["top1_match"][pair],
             "top-3 Jaccard": ag["top3_jaccard"][pair], "Kendall tau": ag["kendall_tau"][pair]}
            for pair in ag["top1_match"]
        ])

    summary = _data(results, "summary_comparison")
    if summary:
        rows = []
        for top_n, by_variant in summary["averages"].items():
            for variant, m in by_variant.items():
                rows.append({"sentences": int(top_n), "summary": variant, **{k: round(v, 4) if v is not None else None
                                                                             for k, v in m.items()}})
        tables["T7_summarization"] = pd.DataFrame(rows)
    return tables


TABLE_TITLES = {
    "T1_dataset": "T1. Dataset",
    "T2_model_comparison": "T2. Model comparison (shared 20% test split)",
    "T3_cross_validation": "T3. 5-fold cross-validation (mean ± std)",
    "T4_significance": "T4. McNemar significance tests between models",
    "T5_ablation": "T5. Ablation: Random Forest F1 when one feature is removed (5-fold CV)",
    "T6_explanation_agreement": "T6. Agreement between explanation methods",
    "T7_summarization": "T7. Summarization: standard vs credibility-aware (averaged over categories)",
}


def to_markdown_table(df: pd.DataFrame) -> str:
    """Plain GitHub-flavoured Markdown table (no tabulate dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v).replace("|", "\\|")
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)
