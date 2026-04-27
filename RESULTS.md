# Pipeline Results — Numerical Summary

Numbers below come directly from the parquet files in [`data/intermediate/`](data/intermediate/). Regenerate the underlying summary at any time with:

```bash
conda run -n base python analysis/summarize_results.py   # writes stage_results.json
```

## Stage 1 — GPU aggregation
- Hourly time-series rows: **9,381,693**
- Daily rows: **406,274**
- Subreddits: **500**
- Coverage: **2023-06-01 → 2024-07-31**
- Comments aggregated: **1,187,443,795**
- Submissions aggregated: **87,224,551**

## Stage 2 — Anomaly detection (rolling 168 h z-score, threshold = 3.0)
- Anomaly windows: **31,938**
- Subreddits with ≥ 1 anomaly: **499 / 500**
- Median peak z-score: **3.67** · 95th percentile: **9.16** · Max: **3,540.33**
- Mean window duration: **1.75 h** (median **1 h**) · mean anomaly hours: **2.38**
- Top-10 most anomaly-prone subreddits: `IndianTeenagers (282)`, `tennis (207)`, `biggboss (204)`, `LoveIslandTV (203)`, `Cricket (203)`, `FapDeciders (198)`, `SquaredCircle (190)`, `ConeHeads (185)`, `naturaltitties (181)`, `futebol (175)`

## Stage 3 — Cross-subreddit propagation (≤ 48 h)
- Connected components (event clusters): **1**
- Member windows in the giant component: **63,876**
- Total cluster duration: **10,246 h**
- Propagation type label: `niche_to_mainstream`
- *Reading:* the 48 h overlap radius and dense top-500 graph collapse to one component — this is a parameterization finding, not an empirical absence of structure. Tightening the radius and adding entity / topic gating is recommended (see report §6).

## Stage 4 — Spike shapes & engagement
- Profiles: **31,938** · shape mix: `double_peak 31,673 (99.2%)`, `slow_burn 196 (0.6%)`, `sharp_spike 64 (0.2%)`, `sustained_plateau 5 (<0.1%)`
- Pearson r(peak z, post-spike avg score) = **−0.003** (p = 0.64) · Spearman r = **0.082** (p = 2.1 × 10⁻⁴⁸)
- Mean engagement ratio (post-spike / baseline) = **1.39** · median = **1.00**
- *Reading:* aggressive `find_peaks` heuristic over-classifies as `double_peak`. Spearman is small but highly significant — bigger spikes do drive *slightly* more engagement on average.

## Stage 5 — Temporal patterns
- Cells: **168** (24 h × 7 days)
- Peak hour-of-day × day-of-week: **Tuesday 16:00 UTC** (520 anomalies)
- Weekday total: **24,603** · weekend total: **7,335** (≈ 5:2 ratio, near uniform per-day rate)

## Stage 6 — Named entity recognition
- Entity rows: **63,614** · unique entities: **36,082** · windows annotated: **1,738** (5.4% of 31,938) · subreddits: **426**
- Label mix: `PERSON 35,314`, `ORG 20,441`, `GPE 7,859`
- Notable real-world entities surfaced: `Hamas (1,050 mentions)`, `Israel (1,330)`, `Gaza (448)`, `Trump (348)`, `Reddit (403)`, `IDF (191)`, `OpenAI`, `NFL`, `Manchester United`
- Many gaming-subreddit-specific tokens also surface (`Zekrom`, `Reshiram`, `Sukuna`) — expected given top-volume Reddit communities

## Stage 7 — Sentiment (cardiffnlp Twitter RoBERTa)
- Windows scored: **2** (partial run; full sweep aborted before completion — `sentiment_partial_saved.parquet`)
- Mean sentiment shift (anomaly − baseline): **+0.023** (not statistically significant)
- Class proportions during anomaly: pos **0.21** · neu **0.62** · neg **0.17**
- Class proportions during baseline: pos **0.24** · neg **0.24**

## Stage 8 — BERTopic
- Windows topic-tagged: **50** · subreddits: **2** (BERTopic was run on a small probe subset)
- Topic 0 representative words: `yeah, bye, baby, baby yeah, yeah yeah, bye baby, image, ama` (74 docs)
- Topic 1: `rule, rule rule, rules, edit meee, signalis rules, signalis` (71 docs)
- Noise label `−1`: **48 / 50 windows** (limited corpus)

## Stage 9 — Event classification (cuML RF + XGBoost, LOOCV)
- Predictions: **500**
- Predicted-class distribution: `breaking_news = 500` (degenerate — all rows assigned the majority class given the small ground-truth sample)
- *Reading:* with 35 labeled events and class imbalance, the classifier defaults to the majority class. Augmenting ground truth and adding class weights is the next step.

## Stage 10 — Sustain / decay (cuML RF, first 4 h features)
- Predictions: **500**
- Ground-truth labels: `decayed 493`, `sustained 7` (extreme imbalance)
- Predicted: `decayed 494`, `sustained 6`
- **Model AUC: 0.949** · precision (positive class): **0.0** · recall: **0.0** · F1: **0.0**
- Top features by importance: `velocity (0.18)`, `early_mean_score (0.14)`, `spike_ratio (0.13)`, `author_diversity (0.12)`, `score_acceleration (0.11)`
- *Reading:* AUC is misleading here because of the imbalance — 6 predicted positives, none correct. The ranking is informative but a positive-class threshold or class-weighted training is needed before deploying.

## Stage 11 — Event forecasting (5-fold stratified, 48 h precursor) ⭐
- Examples: **138** (33 positive ground-truth events, 105 sampled negatives, 3:1 ratio after deduplication)
- **AUC: 0.955** · **Average precision: 0.909** · precision **0.852** · recall **0.697** · F1 **0.767**
- Per-fold AUC: `0.980, 0.986, 0.952, 0.937, 0.952`
- Per-fold F1: `0.77, 0.77, 0.73, 0.73, 0.83`
- Top features: `n_hours_data (0.26)`, `author_concentration (0.10)`, `mean_score (0.08)`, `volume_ratio (0.08)`, `mean_activity (0.08)`, `peak_activity (0.05)`, `max_z_score (0.05)`
- *Reading:* this is the strongest empirical result. With only 33 positives the absolute number is fragile, but the held-out folds are remarkably consistent. Author concentration (how few authors drive the precursor activity) and the gross activity volume / score together carry most of the predictive signal.

---

The figures referenced from `FINAL_REPORT.md` are saved in [`website/figures/`](website/figures/) and are entirely rebuildable from `data/intermediate/` via the two scripts in `analysis/`.
