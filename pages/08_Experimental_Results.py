"""
Experimental Results: every experiment in one place -- headline numbers,
auto-generated key findings, the report's master tables (T1-T7) with
CSV / Markdown / LaTeX downloads, the report figures (F1-F10), and a
reproducibility panel showing where each number came from.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import streamlit as st

from results_io import list_results
from results_summary import (load_all_results, headline_numbers, key_findings, master_tables,
                             to_markdown_table, TABLE_TITLES, RESULT_NAMES)
from report_figures import report_figures

st.set_page_config(page_title="Experimental Results", layout="wide")
st.title("Experimental Results")

results = load_all_results()
missing = [n for n in RESULT_NAMES if results[n] is None]
if len(missing) == len(RESULT_NAMES):
    st.info("No results yet. Generate everything with:\n\n```\npython experiments/run_all.py\n```")
    st.stop()
if missing:
    st.warning("Missing results: " + ", ".join(missing) + ". Run `python experiments/run_all.py` to generate them.")

metas = list_results()
commits = sorted({m["git_commit"] for m in metas})
st.caption(f"{len(metas)} result files · latest generated {max(m['generated_at'] for m in metas)} · "
           f"commit{'s' if len(commits) > 1 else ''} {', '.join(commits)}")

# --- headline numbers ---
h = headline_numbers(results)
k = st.columns(5)
if "best_f1" in h:
    k[0].metric("Best model F1", f"{h['best_f1']:.3f}", help=h["best_model"])
    k[0].caption(h["best_model"])
    k[1].metric("Deployed RF F1", f"{h['deployed_f1']:.3f}")
    k[1].caption("explainable, 6 features")
if "contamination_ours" in h:
    k[2].metric("Fake contamination in summaries", f"{h['contamination_ours']:.0%}",
                f"{(h['contamination_ours'] - h['contamination_standard']) * 100:+.0f} pp vs standard",
                delta_color="inverse")
if "top_feature" in h:
    k[3].metric("Most important feature", h["top_feature"], f"{h['top_feature_delta_f1']:+.3f} F1 if removed",
                delta_color="off")
if "shap_zscore_agreement" in h:
    k[4].metric("SHAP vs z-score agreement", f"{h['shap_zscore_agreement']:.0%}",
                help="Share of reviews where both methods pick the same most important feature")

tabs = st.tabs(["Summary", "Tables", "Figures", "Reproducibility"])
figures = report_figures(results)

with tabs[0]:
    st.subheader("Key findings")
    st.markdown("\n".join(f"- {f}" for f in key_findings(h)))
    st.caption("Generated from the results files, so they update when the experiments are re-run.")
    for name in ["F2_leaderboard", "F6_ablation", "F9_contamination"]:
        if name in figures:
            st.plotly_chart(figures[name][1], use_container_width=True, key=f"summary_{name}")

with tabs[1]:
    tables = master_tables(results)
    if tables:
        name = st.selectbox("Table", list(tables), format_func=lambda n: TABLE_TITLES.get(n, n))
        df = tables[name]
        st.dataframe(df, use_container_width=True, hide_index=True)
        d = st.columns(3)
        d[0].download_button("⬇ CSV", df.to_csv(index=False), f"{name}.csv", "text/csv")
        d[1].download_button("⬇ Markdown", to_markdown_table(df), f"{name}.md", "text/markdown")
        d[2].download_button("⬇ LaTeX", df.to_latex(index=False, float_format="%.4f",
                                                   caption=TABLE_TITLES.get(name, name)),
                             f"{name}.tex", "text/plain")
        st.caption("All tables are also written to `results/tables/` and `docs/RESULTS.md` by "
                   "`python experiments/export_tables.py`.")

with tabs[2]:
    if st.button("Export all figures to docs/figures/ (PNG)"):
        os.makedirs("docs/figures", exist_ok=True)
        with st.spinner("Rendering PNGs..."):
            for fname, (_, fig) in figures.items():
                fig.write_image(f"docs/figures/{fname}.png", scale=2)
        st.success(f"Wrote {len(figures)} figures to docs/figures/.")
    cols = st.columns(2)
    for i, (fname, (caption, fig)) in enumerate(figures.items()):
        with cols[i % 2]:
            st.markdown(f"**{fname.split('_')[0]}.** {caption}")
            st.plotly_chart(fig, use_container_width=True, key=f"fig_{fname}")
            png = f"docs/figures/{fname}.png"
            if os.path.exists(png):
                with open(png, "rb") as f:
                    st.download_button("⬇ PNG", f.read(), f"{fname}.png", "image/png", key=f"dl_{fname}")

with tabs[3]:
    table = pd.DataFrame(metas)[["name", "generated_at", "git_commit", "dataset", "n_rows", "script"]]
    st.dataframe(table, use_container_width=True, hide_index=True)
    datasets = {m["dataset"] for m in metas}
    if len(datasets) > 1:
        st.error(f"Results come from different datasets ({', '.join(sorted(datasets))}); the tables don't belong together.")
    if len(commits) > 1:
        st.warning("Results were generated at different commits. That's fine while developing, but re-run "
                   "everything before the final report so all numbers come from the same code.")
    else:
        st.success("All results come from the same commit.")
    st.markdown("**Reproduce every number:**")
    st.code("pip install -r requirements.txt\n"
            "python experiments/run_all.py          # ~25 min on a laptop; writes results/, docs/RESULTS.md, docs/figures/\n"
            "# DistilBERT: run notebooks/bert_finetune_colab.ipynb on Colab, put distilbert.csv +\n"
            "# distilbert_meta.json in results/predictions/, then re-run run_all.py", language="bash")
