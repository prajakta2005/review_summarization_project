# System Output — Screenshots

Captured automatically by `python scripts/capture_screenshots.py` (Streamlit + Playwright,
1440 px wide, full Kaggle dataset of 40,412 reviews selected on Home). Re-run the script after
any UI change to refresh them all. Live Demo examples are drawn at random from the held-out
test split, so the reviews differ between runs.

**For the report's "System Output" section, use S2, S3 and S5.** They show the three things the
brief asks for: credibility scores, explanations and generated summaries.

## S1. Home: overview
![Home](screenshots/01_home.png)
Overview of the full dataset: review count, average credibility, High/Low counts, the credibility
distribution and global feature importance (mean |SHAP| or Random Forest impurity).

## S2. Live Demo: a genuine review
![Live Demo, genuine example](screenshots/02_live_demo_genuine.png)
A held-out genuine review scored by the system: credibility gauge, SHAP waterfall showing how each
signal moved the score, the factors in plain English, the words that mattered (LIME), and how
much weight the review would get in a summary.

## S3. Live Demo: a fake review
![Live Demo, fake example](screenshots/03_live_demo_fake.png)
A held-out computer-generated review. Repetitive wording pulls its credibility far below the
average, it falls under the low-credibility flags, and its summary weight is small.

## S4. Explainability: one review explained three ways
![Explainability](screenshots/04_explainability_waterfall.png)
SHAP waterfall and LIME weights side by side, with the SHAP-based and original z-score
explanations for comparison.

## S5. Summarization: standard vs credibility-aware
![Summarization](screenshots/05_summarization_side_by_side.png)
The same extractive summarizer with and without credibility weighting. Each sentence shows its
source review, credibility score and true label. Below that: the full-dataset evaluation (fake
contamination 92% → 42%, ROUGE-L 0.314 → 0.405).

## S6. Review Explorer
![Review Explorer](screenshots/06_review_explorer.png)
Product-level summary with low-credibility flags, attribute consensus/conflicts and a per-review drilldown.

## S7. Model Comparison
![Model Comparison](screenshots/07_model_comparison.png)

## S8. Performance Results
![Performance](screenshots/08_performance.png)

## S9. Ablation Study
![Ablation Study](screenshots/09_ablation.png)

## S10. Experimental Results
![Experimental Results](screenshots/10_experimental_results.png)

## S11. Dataset Explorer
![Dataset](screenshots/11_dataset.png)

## S12. Novelty
![Novelty](screenshots/12_novelty.png)
