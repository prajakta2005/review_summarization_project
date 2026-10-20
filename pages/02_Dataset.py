"""
Dataset Explorer: size, class balance, categories, ratings, text length
and feature distributions for genuine vs fake reviews.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd

from dashboard import require_pipeline, show_result_or_hint
from dataset_profile import profile_dataset, label_groups
from features import FEATURE_COLUMNS
from charts import dataset as charts

st.set_page_config(page_title="Dataset", layout="wide")
st.title("Dataset Explorer")

res = require_pipeline()
df = res.df


@st.cache_data(show_spinner=False)
def live_profile(cache_key, _df):
    return profile_dataset(_df)


cache_key = (st.session_state.get("data_source"), st.session_state.get("max_rows"),
             st.session_state.get("upload_key"), len(df))
profile = live_profile(cache_key, df)

# --- headline numbers ---
m = st.columns(5)
m[0].metric("Reviews", f"{profile['n_after_preprocess']:,}")
counts = profile["label_counts"]
if profile["labeled"]:
    m[1].metric("Genuine", f"{counts.get('genuine', 0):,}")
    m[2].metric("Fake", f"{counts.get('fake', 0):,}")
else:
    m[1].metric("Genuine", "n/a")
    m[2].metric("Fake", "n/a")
m[3].metric("Categories / products", profile["n_categories"])
m[4].metric("Duplicates removed", profile["duplicates_removed"])

if not profile["labeled"]:
    st.warning("This dataset has no ground-truth labels, so genuine/fake breakdowns aren't available. "
               "Pick the **Kaggle fake-reviews dataset** on Home, or see the full-dataset reference tab.")

tabs = st.tabs(["Overview", "Categories", "Text & Ratings", "Feature Distributions",
                "Raw Data", "Full-dataset reference"])

with tabs[0]:
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.class_balance_bar(profile), use_container_width=True)
    c2.plotly_chart(charts.category_donut(profile), use_container_width=True)
    if profile["labeled"]:
        st.caption("Labels: **genuine** = `OR` (original, human-written), "
                   "**fake** = `CG` (computer-generated). In this dataset `product_id` is the Amazon category, "
                   "so product-level summaries are category-level summaries.")

with tabs[1]:
    st.plotly_chart(charts.category_stacked_bar(profile), use_container_width=True)
    st.dataframe(pd.DataFrame(profile["categories"]), use_container_width=True, hide_index=True)

with tabs[2]:
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.rating_by_label_bar(profile), use_container_width=True)
    c2.plotly_chart(charts.feature_hist(profile, "length"), use_container_width=True)
    st.dataframe(pd.DataFrame(profile["length_by_label"]).T.rename_axis("label"),
                 use_container_width=True)
    st.caption("Length is in words.")

with tabs[3]:
    feature = st.selectbox("Feature", FEATURE_COLUMNS)
    # violins need raw values: sample so the browser stays fast
    sample = df.sample(min(len(df), 5000), random_state=42).assign(label_name=lambda d: label_groups(d))
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.feature_violin(sample, feature), use_container_width=True)
    c2.plotly_chart(charts.feature_hist(profile, feature), use_container_width=True)
    if len(df) > 5000:
        st.caption(f"Violin drawn from a random 5,000-review sample; histogram uses all {len(df):,}.")
    stats = pd.DataFrame(profile["feature_stats_by_label"]).T
    if profile["labeled"]:
        st.plotly_chart(charts.separation_bar(profile), use_container_width=True)
        st.caption("Cohen's d is a quick preview of separation (|d| ≈ 0.2 small, 0.5 medium, 0.8 large). "
                   "The Ablation Study measures what each feature actually contributes to the model.")
    st.dataframe(stats, use_container_width=True)

with tabs[4]:
    view = df.assign(label=label_groups(df))
    f1, f2, f3 = st.columns(3)
    cat_col = "category" if "category" in view.columns else "product_id"
    cats = f1.multiselect("Category", sorted(view[cat_col].unique()))
    labels = f2.multiselect("Label", sorted(view["label"].unique()))
    ratings = f3.multiselect("Rating", sorted(view["rating"].unique()))
    if cats:
        view = view[view[cat_col].isin(cats)]
    if labels:
        view = view[view["label"].isin(labels)]
    if ratings:
        view = view[view["rating"].isin(ratings)]
    cols = ["review_id", cat_col, "rating", "label", "review_text"] + FEATURE_COLUMNS
    st.caption(f"{len(view):,} reviews match (showing up to 1,000).")
    st.dataframe(view[cols].head(1000), use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", view[cols].to_csv(index=False),
                       file_name="reviews_filtered.csv", mime="text/csv")

with tabs[5]:
    st.write("Profile of the **complete** Kaggle dataset, generated offline, "
             "so it's available whatever data source is loaded.")
    saved = show_result_or_hint("dataset_profile", "experiments/run_dataset_profile.py")
    if saved:
        full = saved["data"]
        st.caption(f"Generated {saved['meta']['generated_at']} · commit {saved['meta']['git_commit']}")
        k = st.columns(4)
        k[0].metric("Raw rows", f"{full['n_raw']:,}")
        k[1].metric("After preprocessing", f"{full['n_after_preprocess']:,}")
        k[2].metric("Genuine / fake", f"{full['label_counts']['genuine']:,} / {full['label_counts']['fake']:,}")
        k[3].metric("Categories", full["n_categories"])
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.category_stacked_bar(full), use_container_width=True, key="full_cat")
        c2.plotly_chart(charts.separation_bar(full), use_container_width=True, key="full_sep")
