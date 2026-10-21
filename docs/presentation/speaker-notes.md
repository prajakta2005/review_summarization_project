# Speaker Notes

Slide-by-slide script for the second-review deck (*Explainable Review Intelligence*, 19 slides).
For every slide: **who** presents, **how long**, the **opening line** (say it word for word to get
started), the **points** to make, what to **point at**, and the **hand-off** to the next slide.

- **Member A**: recap, models, performance, ablation, objectives, advantages/limitations, future scope.
- **Member B**: summarization results, explainability, comparison, novelty, live demo.
- **Total:** about 15 minutes of slides + 5 minutes of live demo. Times in brackets are targets; if you
  run long, shorten the slides marked *(can trim)*.
- Numbers are from [`docs/RESULTS.md`](../RESULTS.md). Don't round them differently from the slides.
- Likely questions and answers: [demo-runbook.md](demo-runbook.md#questions-youll-probably-get).

---

## Part 1 · Recap of the first review (≈ 1.5 min)

### Slide 1 · Cover (A · 20 s)

**Start with:** "Good morning. We're [Member A] and [Member B], and our project is *Explainable Review
Intelligence*."

- One sentence on what it does: we score how credible each review is, explain why, and use that score
  to build summaries that fake reviews can't dominate.
- Say what this review covers: "The first review covered the problem and the design, so we'll recap
  that in one slide and spend most of our time on results, comparison with existing work, novelty and
  what we achieved."

**Hand-off:** "First, a quick reminder of the problem."

### Slide 2 · Recap: the problem and our pipeline (A · 1 min)

**Start with:** "Online shoppers rely on review summaries, but those summaries treat every review as
equally trustworthy, including fake ones."

- The idea in one line: credibility should decide how much a review counts in a summary.
- Walk the seven boxes left to right in one breath. **Point at the two blue boxes** (credibility score and
  weighted summary): "these two are where our system differs from a normal summarizer."
- Read the last line quickly: since the first review we moved to the full 40,000-review dataset, compared
  nine classifiers, added SHAP and LIME, evaluated the summaries, ran an ablation study, made everything
  reproducible, and built a live demo.

**Hand-off:** "Here's the one number that sums up the results." *(B takes over.)*

---

## Part 2 · Results and analysis (≈ 6 min)

### Slide 3 · Headline result: 92% → 42% (B · 40 s)

**Start with:** "If you remember one number from this talk, make it this one."

- Pause on the number. Then explain it: "With a normal summarizer, 92% of the summary sentences came from
  fake reviews. With our credibility weighting, 42%."
- Stress fairness: same summarizer, same reviews; the only difference is the credibility weighting.
- Scale: 40,412 labelled reviews, 10 product categories, 5-sentence summaries.

**Hand-off:** "Let's look at where that number comes from."

### Slide 4 · Fake content halves at every summary length (B · 45 s)

**Start with:** "Orange is the standard summarizer, blue is ours."

- **Point at the three pairs of bars:** 97% → 43% at 3 sentences, 92% → 42% at 5, 91% → 36% at 8. The
  effect doesn't depend on summary length.
- **Point at the right column:** contamination down 50 percentage points; ROUGE-L against a genuine-only
  summary up from 0.31 to 0.41; the reviews we quote are far more credible (39 → 71).
- Pre-empt the leakage question: "Every review's score comes from a model that never saw that review:
  5-fold out-of-fold predictions."

**Hand-off:** "Is this an average hiding big differences? Per category:"

### Slide 5 · The drop holds in 9 of 10 categories (B · 40 s) *(can trim)*

**Start with:** "Each line is one product category: orange dot is the standard summary, blue is ours."

- Best cases: Home & Kitchen and Kindle Store go from 80% to 0%.
- Be upfront about the exception: Clothing, Shoes & Jewelry stays at 80%. Its fake sentences still score
  high enough to be chosen; better credibility scores are what would move it.
- **Why standard summaries are so contaminated** (this is an insight, say it clearly): computer-generated
  reviews use generic, typical wording, and a summarizer that picks the "most central" sentences rewards
  exactly that.

**Hand-off:** "Fewer fake sentences is good, but are the summaries actually better?"

### Slide 6 · Closer to genuine reviewers, without deleting anything (B · 50 s)

**Start with:** "On the left, we measure how close each summary is to a summary built only from genuine
reviews."

- **Left chart:** ours is closer on every ROUGE measure; ROUGE-2 almost doubles, 0.16 → 0.31.
- **Right chart:** "We also asked: why not just delete low-credibility reviews?" Deleting everything below
  60 gets contamination down to 20%, but ROUGE-L drops to 0.32 because genuine content gets thrown away.
- Design point: we down-weight suspicious reviews and list them separately as flags, so nothing is
  hidden from the user.

**Hand-off:** "All of this depends on the credibility model, so how good is it?" *(A takes over.)*

### Slide 7 · Nine classifiers, one shared test set (A · 45 s)

**Start with:** "We trained nine classifiers, Random Forest, Logistic Regression and SVM, on three
kinds of input, all tested on exactly the same held-out reviews."

- Inputs: our 6 credibility features, TF-IDF text, and both together. 5-fold cross-validation.
- **Point at the top bar:** best is a linear SVM on text plus features, F1 0.951.
- **Point at the blue bars:** the interpretable models. Our deployed Random Forest is 0.763.
- DistilBERT: notebook is ready and plugs into the same comparison; it needs a GPU run, so it's pending.

**Hand-off:** "So why do we deploy the 0.76 model and not the 0.95 one?"

### Slide 8 · We trade 0.19 F1 for scores a person can check (A · 50 s)

**Start with:** "This is the key design decision in the project, and we measured it rather than hiding it."

- Text models know *that* a review is fake; they can't say *why* in terms a shopper understands. Ours
  explains every score in six signals, and those explanations drive the summaries.
- **Point at the two clusters:** text models top right, interpretable models below; our deployed model
  is "RF (deployed)".
- The surprising finding: adding our six features to a text model does **not** improve it significantly
  (McNemar p = 0.84 and 0.13). Their value is explanation, not extra accuracy.
- Future-proof: a stronger detector's scores could drive the weighting while our model keeps explaining.

**Hand-off:** "Here's where the deployed model goes wrong."

### Slide 9 · The deployed model lets 31% of fakes through (A · 45 s)

**Start with:** "These are confusion matrices on the same 8,083 test reviews: our model on the left,
the best model on the right."

- **Point at the top-right cell of the left matrix:** 31% of fake reviews are passed as genuine.
- Connect it to the summaries: "That's why the summaries are still 42% contaminated. Those fakes keep
  high weight. A better detector would directly reduce contamination."
- **Point at the table:** accuracy 0.756, precision 0.728, recall 0.819, F1 0.771, ROC-AUC 0.822, against
  0.95+ for the best SVM.
- Rigour: 33 of 36 model pairs differ significantly (McNemar test).

**Hand-off:** "Which of our six features is the model actually using?"

### Slide 10 · Two signals carry the model (A · 45 s)

**Start with:** "We retrained the model six times, each time leaving one feature out."

- **Point at the two long orange bars:** without repetition ratio F1 drops by 0.192; without length by
  0.106.
- The other four change F1 by less than 0.01.
- Interpretation: computer-generated reviews repeat words, and that one signal explains most of the
  model's skill. SHAP, permutation and impurity importance all agree on the top two.

**Hand-off:** "Which brings us to explanations. Can we trust them?" *(B takes over.)*

### Slide 11 · We tested the explanations, then switched to SHAP (B · 55 s)

**Start with:** "Showing an explanation isn't enough. We wanted to know whether it's actually true to
the model."

- **Left chart (deletion test):** reset the feature an explanation calls most important and see how much
  the prediction moves. SHAP moves it most: 0.101, three times a random feature (0.032). LIME 0.094,
  our original z-score method 0.062.
- **Right chart:** our original explanations agreed with SHAP on the top reason for only 32% of reviews:
  they described what was *unusual* about a review, not what the model *used*.
- So the system now explains every score with SHAP, and LIME adds word-level highlights.

**Hand-off:** "How does all this compare with existing approaches?"

---

## Part 3 · Comparison (≈ 2 min)

### Slide 12 · Comparison matrix (B · 50 s)

**Start with:** "Existing work covers this problem one piece at a time."

- Read down the first column: fake-review detection labels reviews; review summarization condenses them
  but trusts all of them; explainable ML explains models in general.
- **Point at the blue row:** this system covers all six capabilities, including the last three that none of
  the others have: a trust-weighted summary, sentence-level traceability, and measuring fake content in
  the output.
- If asked: the rows are representative families; [say the closest combined system you found in the
  literature search, and how yours differs].

**Hand-off:** "What does that mean for someone actually reading a summary?"

### Slide 13 · Comparison with existing systems (B · 1 min)

**Start with:** "Row by row, here's how a user's experience changes, each with the number that backs it."

- **Trust:** existing summarizers count every review equally; we weight by an explained score, 92% → 42%.
- **Content:** closer to what genuine reviewers say, ROUGE-L 0.31 → 0.41.
- **Suspicious reviews:** not silently kept or deleted; down-weighted and listed as flags.
- **Why a score:** per-review SHAP reasons, three times more faithful than random.
- **Verifiability:** any summary sentence traces back to its review, score and reasons (you'll see it in
  the demo).
- **Evaluation:** we measure accuracy *and* summary contamination *and* explanation faithfulness, and all
  of it reproduces with one command.

**Hand-off:** "So what exactly is new?"

---

## Part 4 · Novelty and objectives (≈ 3 min)

### Slide 14 · The novelty is the coupling, and we measured it (B · 1 min 15 s)

**Start with:** "Detection, explanation and summarization each exist on their own. Our contribution is
connecting them, and measuring what that connection buys."

Go card by card, one sentence each:
1. **Credibility-gated summaries:** the score decides how much each review counts; fake content 92% → 42%.
2. **End-to-end traceability:** every summary sentence links to its review, its score and its SHAP reasons.
3. **Validated explanations:** we compared three methods and chose SHAP because it's the most faithful.
4. **A quantified trade-off:** 0.19 F1 given up for explainability, and shown not to matter on top of text.
5. **A contamination metric:** a simple way to evaluate trust-aware summaries wherever fake labels exist.

Avoid the word "first". Say "combined and measured together, unlike the single-purpose systems in the
matrix."

**Hand-off:** "Against the objectives we set for this phase:" *(A takes over.)*

### Slide 15 · Objectives achieved (A · 1 min)

**Start with:** "We set nine objectives for this phase. Eight are fully met; one is waiting on a GPU run."

- Don't read every row. Pick three:
  - **Full dataset:** 40,412 labelled reviews with one shared split.
  - **Summary comparison:** contamination 92% → 42%.
  - **Experimental results:** 7 tables, 13 charts, all reproducible with one command.
- Be direct about the partial one: DistilBERT. The notebook is ready and merges in automatically; it only
  needs a GPU (Colab, about 20 minutes).

**Hand-off:** "What the system does well, and where it falls short."

---

## Part 5 · Assessment and close (≈ 2.5 min + demo)

### Slide 16 · Advantages and limitations (A · 1 min)

**Start with:** "We'd rather point out the limitations ourselves."

- **Advantages (left):** halves fake content; every score explained in plain English; every summary
  sentence traceable; extractive, so nothing is invented; suspicious reviews flagged, not hidden; fast
  (40,000 reviews in about 12 seconds on a laptop); fully reproducible.
- **Limitations (right):** say all of them, calmly:
  - 42% of summary sentences still come from fakes: halved, not solved.
  - The explainable model trails text models by 0.19 F1.
  - "Fake" here means computer-generated; human-written fakes may behave differently.
  - The dataset has categories, not individual products.
  - ROUGE against a genuine-only summary is a proxy, not a human judgement.
  - DistilBERT results pending.

**Hand-off:** "Several of these limitations point straight to future work."

### Slide 17 · Future scope (A · 45 s) *(can trim)*

**Start with:** "The biggest improvement follows directly from our results."

- **Better credibility scores:** contamination is limited by the detector, so let a stronger text or BERT
  model drive the weights while SHAP on the six features keeps the explanations.
- Then, briefly: finish DistilBERT; test on human-written deceptive reviews; add reviewer-level signals
  like posting bursts and duplicate text across accounts; abstractive summaries that only use
  high-credibility sentences and cite them; a small blind human evaluation.

**Hand-off:** "Let us show you the system working." *(B goes to the laptop.)*

### Slide 18 · Live demo (B drives, A narrates · 5 min)

**Start with (A):** "[Member B] will score a review the model has never seen, show why it got that score,
and then show how the summary changes."

Switch to the dashboard (already open, Kaggle dataset selected, pages pre-loaded). Follow
[demo-runbook.md](demo-runbook.md#the-6-minute-demo-script):

1. **Live Demo → 🔴 Fake example:** low score; the big red bar in the waterfall is repetition.
2. **🟢 Genuine example:** high score; specific details push it up; high weight in a summary.
3. Type a hype review (*"BEST product EVER!!! love love love it, buy buy buy!!!"*, 5 stars): the score
   and reasons update instantly.
4. **Summarization → Books_5, 5 sentences:** count the FAKE badges on the left vs the right.
5. **Novelty → Trace a summary sentence:** pick one; review, score, true label, SHAP reasons.

If the app misbehaves, stay on this slide: the two screenshots are the fallback. Say "here's the same
screen we would have shown" and walk through them.

**Hand-off (A):** "To wrap up:"

### Slide 19 · Conclusion · Thank you (both · 30 s)

**Start with (A):** "Three takeaways."

1. Credibility weighting halves fake content in review summaries, 92% → 42%.
2. Every score is explained, and we tested that the explanations are faithful to the model.
3. Every summary sentence can be traced back to its review and its reasons.

**Close (B):** "Thank you. We're happy to take questions."

Keep the dashboard open in the background for questions. The most likely ones (why not BERT, is 42%
still high, did the model see the reviews it scored, what "fake" means) are answered in
[demo-runbook.md](demo-runbook.md#questions-youll-probably-get).

---

## Timing summary

| Part | Slides | Presenter | Target |
|---|---|---|---|
| Recap | 1–2 | A | 1.5 min |
| Results: summaries | 3–6 | B | 3 min |
| Results: models, performance, ablation | 7–10 | A | 3 min |
| Results: explainability | 11 | B | 1 min |
| Comparison | 12–13 | B | 2 min |
| Novelty | 14 | B | 1.25 min |
| Objectives, advantages/limitations, future | 15–17 | A | 2.75 min |
| Live demo | 18 | B + A | 5 min |
| Conclusion | 19 | both | 0.5 min |

**Rehearse the hand-offs:** each one is written as the last line of the previous slide, so the next
speaker can start without a pause.
