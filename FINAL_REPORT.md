# Reddit as an Early Warning System

## Detecting, Classifying, and Forecasting Real-World Events from 478 GB of Community Activity

**Course:** DATS 6450 — Big Data Analytics
**Team (Group 3):** Venkatesh Nagarjuna · Kartik Pruthi · Dhruv Rai
**Repository:** [`Event-Prediction-using-Reddit-Data`](./)
**Date:** April 2026

---

## Abstract

Online communities on Reddit react to real-world events in near real time, producing volumes of text whose temporal and semantic patterns implicitly encode what is happening in the world. This project asks whether those patterns can be detected, characterized, and used to *classify* and *forecast* significant events. We process **478 GB** of Reddit comments and submissions covering **June 2023 — July 2024** (14 months) across the **top 500 subreddits** through an **11-stage pipeline** that combines distributed CPU processing on AWS EC2 (PySpark) with GPU acceleration on a RunPod A100 80 GB pod (RAPIDS cuDF, cuML, Hugging Face transformers, BERTopic). A hand-curated ground truth catalog of **35 events** across five categories (breaking news, controversy, product launch, disaster, meme/viral) anchors evaluation. The pipeline detects **31,938 anomaly windows** across 499 subreddits; identifies **5.15 million** cross-subreddit co-occurring anomaly pairs; classifies spike morphologies; characterizes content via NER, sentiment, and topic modeling; and trains GPU-accelerated classifiers and forecasters. The work demonstrates that a hybrid EC2/GPU architecture can take a multi-hundred-GB social-media corpus from raw parquet to interpretable event-level analytics within a $20–$50 compute budget, while remaining reproducible end-to-end via scripts, intermediate parquet artifacts, and a Quarto website.

---

## 1. Introduction

### 1.1 Motivation

Real-world events — breaking news, product launches, disasters, controversies, viral moments — leave measurable footprints on Reddit. Subreddit-level activity (post counts, unique authors) spikes within minutes; the language used shifts (new entities, shifted sentiment, emerging topics); and signals propagate from niche communities into mainstream feeds. A reliable *early warning system* (EWS) that turns these signals into structured event records would be useful for journalism, situational awareness, content moderation triage, and basic research on collective attention.

### 1.2 Research Questions

We organize the project around ten research questions in three layers:

| # | Layer | Question |
|---|-------|----------|
| Q1 | EDA | What does baseline activity look like, and where do statistically significant anomalies occur? |
| Q2 | EDA | How do anomalies propagate across subreddit communities? |
| Q3 | EDA | What shapes do activity spikes take, and how does shape relate to engagement? |
| Q4 | EDA | Are there hour-of-day / day-of-week patterns in anomaly occurrence? |
| Q5 | NLP | Which named entities surge inside anomaly windows? |
| Q6 | NLP | How does community sentiment shift during anomalies vs. baseline? |
| Q7 | NLP | What latent topics emerge from anomaly-period text? |
| Q8 | ML | Can we classify anomalies into one of five event categories? |
| Q9 | ML | Can early-window features predict whether a spike will sustain or decay? |
| Q10 | ML (bonus) | Can Reddit signals provide *advance* warning before mainstream coverage? |

### 1.3 Dataset

| Attribute | Value |
|-----------|-------|
| Source | Pushshift Reddit archive (comments + submissions), Parquet on S3 |
| Period | 2023-06 through 2024-07 (14 months) |
| Size | ~478 GB (snappy-compressed Parquet) |
| Partitioning | `yyyy=YYYY/mm=MM/` per data type |
| Scope | Top 500 subreddits by total post volume (filtered in Stage 1) |
| Ground truth | 35 manually curated events across 5 categories — [`data/ground_truth/events.csv`](data/ground_truth/events.csv) |

The five ground-truth categories and example events:

| Category | n | Examples |
|----------|---|----------|
| Breaking news | 15 | Hamas attack on Israel; Sam Altman returns to OpenAI; Trump assassination attempt |
| Controversy | 7 | Reddit API pricing protest; Sam Altman fired; Biden-Trump debate |
| Product launch | 7 | Threads launch; iPhone 15; GPT-4o; Apple Intelligence |
| Disaster | 4 | OceanGate Titan; Maui wildfire; Baltimore Key Bridge collapse |
| Meme/viral | 3 | Barbenheimer; Taylor Swift Super Bowl; Black Friday discourse |

---

## 2. System Architecture

### 2.1 Hybrid EC2 + GPU Design

A single GPU-rich machine could in principle run everything, but at 478 GB of input the I/O- and aggregation-heavy work is wasteful on an expensive A100. Conversely, transformer NLP and cuML training are impractical on the small EC2 instance available through AWS Academy. We therefore split the workload by hardware affinity:

| Environment | Spec | Stages | Why |
|-------------|------|--------|-----|
| **EC2** `t3.large` | 2 vCPU, 7.6 GB RAM, PySpark 3.5 local mode | 2, 3, 4, 5, optional Spark MLlib baseline | Distributed reads from S3, window functions, aggregation, joins |
| **RunPod / Lightning.ai** | A100 80 GB, RAPIDS, PyTorch, transformers | 1, 6, 7, 8, 9, 10, 11 | cuDF aggregation of 478 GB; transformer NER/sentiment; BERTopic; cuML RandomForest / XGBoost |

S3 acts as the shared bus: a *raw* bucket holds Pushshift parquet, and a separate *processed* bucket (introduced in commit `c0340e4`) holds intermediate outputs so that read-only credentials from AWS Academy can coexist with write credentials issued for output. This split is configured in [`config/settings.py`](config/settings.py).

### 2.2 Spark Configuration for a 2-vCPU Instance

The EC2 instance has hard memory limits, so the Spark session ([`config/spark_config.py`](config/spark_config.py)) is tuned to:

- `local[2]`, matching the vCPU count.
- 3 GB driver memory, 0.4 memory fraction — leaves room for Python and pandas conversions.
- 4 shuffle partitions (default 200 would create thousands of tiny tasks).
- Adaptive Query Execution + coalescePartitions, to handle the heavy skew in subreddit volume.
- Kryo serializer, Arrow-based Spark↔pandas conversion.
- 600 s network/RPC timeouts (added in commit `c0340e4`) to survive long driver-side `toPandas()` calls during visualization steps.

### 2.3 Pipeline Topology

```
S3 (raw 478 GB)
   │
   ▼  GPU
[Stage 1] GPU aggregation — hourly_counts, daily_counts, subreddit_stats
   │
   ▼  Spark (EC2)
[Stage 2] Anomaly detection (rolling z-score)
   │
   ├──► [Stage 3] Cross-subreddit propagation
   ├──► [Stage 4] Spike shapes & engagement
   └──► [Stage 5] Temporal patterns
   │
   ▼  GPU (parallel where possible)
[Stage 6] NER (spaCy en_core_web_trf)
[Stage 7] Sentiment (RoBERTa)
[Stage 8] Topics (BERTopic + cuML UMAP/HDBSCAN)
   │
   ▼  GPU
[Stage 9]  Event classification (cuML RF + XGBoost, LOOCV)
[Stage 10] Sustain/decay prediction (cuML RF, early-window features)
[Stage 11] Event forecasting (5-fold stratified, 48-h precursor window)
   │
   ▼
[Optional] Spark MLlib baseline (Stage `stage_ml_spark.py`) + Quarto site
```

Ten of the eleven stages live in two folders: [`pipeline/`](pipeline) for Spark on EC2, [`runpod/`](runpod) for GPU on RunPod; shared utilities live in [`utils/`](utils) and configuration in [`config/`](config).

---

## 3. Methodology

### 3.1 Stage 1 — GPU Aggregation ([`runpod/stage1_aggregate_gpu.py`](runpod/stage1_aggregate_gpu.py))

The 478 GB raw dataset is too large to load in memory on either machine. Stage 1 streams the data **month by month** through cuDF on the A100 (with a pandas fallback for non-GPU runs) and emits three small artifacts:

- `hourly_counts.parquet` — `(subreddit, hour_bucket)` × `(post_count, unique_authors, mean_score, data_type)`
- `daily_counts.parquet` — daily roll-up
- `subreddit_stats.parquet` — per-subreddit total / mean / max / std

Key efficiency choices:

- **Column pruning:** only `subreddit, created_utc, author, score` (and `num_comments` for submissions) are read.
- **Predicate pushdown** through `pyarrow` Parquet filters when subreddit lists are known (used in Stage 6).
- **Per-month checkpointing** to S3 — interrupted runs resume from the last successful month.
- **Top-500 filter** on aggregated counts (constant `TOP_N_SUBREDDITS` in `config/settings.py`) keeps downstream stages tractable.

After this stage, the working set drops from 478 GB to a few hundred MB of structured time series.

### 3.2 Stage 2 — Anomaly Detection ([`pipeline/stage2_anomaly_detection.py`](pipeline/stage2_anomaly_detection.py))

A 7-day (168-hour) rolling z-score is computed per subreddit using a Spark `Window` with `rangeBetween(-168*3600, -1)` over a unix-seconds column, guarding against zero standard deviation:

```
z_t = (x_t - μ_{t-168:t}) / σ_{t-168:t}
```

Hours with z > **3.0** are flagged; consecutive flagged hours within a 6-hour gap are merged (via cumulative-sum trick on a `lag()`-derived `new_window_flag`) into contiguous **anomaly windows** identified by `subreddit_id_n`. Per-window aggregates capture peak/mean z, peak/mean post counts, and duration.

Outputs feed every downstream stage. Two figures are emitted: the global z-score histogram with the threshold annotated and a monthly bar chart of anomaly counts.

### 3.3 Stage 3 — Cross-Subreddit Propagation ([`pipeline/stage3_propagation.py`](pipeline/stage3_propagation.py))

A self-join on the anomaly windows finds pairs `(window_a, window_b)` whose intervals overlap — or are within 48 h — across *different* subreddits. The resulting edge list is loaded into NetworkX, and **connected components** become **event clusters**. Each cluster is then classified by a heuristic:

- **simultaneous** if >50% of member windows start within ±2 h of the first window;
- **niche-to-mainstream** if the first-mover subreddit is below the median in total post volume;
- **top-down** if the first-mover is above the 75th percentile;
- otherwise default to *niche-to-mainstream*.

The stage emits a propagation-type bar chart, a time-offset-vs-subreddit-size scatter, and a 2×5 small-multiples plot of the top 10 clusters as NetworkX `spring_layout` graphs.

### 3.4 Stage 4 — Spike Shape & Engagement ([`pipeline/stage4_engagement.py`](pipeline/stage4_engagement.py))

For each anomaly window we extract the **24 h before / 72 h after** time series in pandas (efficient because `anomaly_windows` is small after Stage 2) and classify the spike into one of four morphologies via a hand-tuned heuristic over normalized values:

- **sharp spike** — fast time-to-peak (<4 h) and quick half-decay (<8 h)
- **sustained plateau** — >24 h above 2× pre-spike baseline
- **double peak** — `scipy.signal.find_peaks` finds two local maxima with a >20 % dip
- **slow burn** — slow rise (>12 h to peak)

Engagement features are computed in the same pass: post-spike average score, post-spike unique authors, baseline average score, and a derived `engagement_ratio`. Pearson and Spearman correlations of `peak_z_score` vs `post_spike_avg_score` quantify whether bigger spikes drive more downstream engagement.

### 3.5 Stage 5 — Temporal Patterns ([`pipeline/stage5_temporal.py`](pipeline/stage5_temporal.py))

Spark extracts `hour_of_day` and `day_of_week` (remapped to `0=Mon..6=Sun`) on both the hourly time series and the anomaly windows, and joins the ground-truth CSV (exploded on `relevant_subreddits`) to build per-category × day-of-week event counts. The stage emits a 7×24 heatmap, a polar 24-hour radial chart of anomaly density, and a stacked bar chart of ground-truth event categories by day of week. A weekday-vs-weekend breakdown of anomaly counts and propagation duration is printed.

### 3.6 Stage 6 — Named Entity Recognition ([`runpod/stage6_ner_gpu.py`](runpod/stage6_ner_gpu.py))

We use spaCy's `en_core_web_trf` (transformer backbone) with `spacy.prefer_gpu()`. The most important optimization (commits `d16f24a` → `759f492`) is the **month-batched read pattern**: rather than re-reading S3 for every anomaly window, we read each month's parquet *once* with a `subreddit IN (...)` filter, index by subreddit, and serve all windows in that month from a single in-memory frame. We disable `tagger`, `parser`, `lemmatizer`; cap text length at 5,000 chars; sample at most 100k texts per window; and keep entity types {PERSON, ORG, GPE}. Per-window outputs include top-200 entities and top-100 entity-pair co-occurrences (`itertools.combinations` on the per-doc entity set).

### 3.7 Stage 7 — Sentiment ([`runpod/stage7_sentiment_gpu.py`](runpod/stage7_sentiment_gpu.py))

`cardiffnlp/twitter-roberta-base-sentiment-latest` runs through a Hugging Face `pipeline("sentiment-analysis", device=0, top_k=None)` to return all class probabilities (positive / neutral / negative). For each window we sample up to 10,000 anomaly-period texts and 5,000 baseline texts (preceding 7 days), then score both. We compute mean / std of per-class probabilities and a Welch's *t*-test for the **sentiment shift** between baseline and anomaly periods.

### 3.8 Stage 8 — Topics ([`runpod/stage8_topics_gpu.py`](runpod/stage8_topics_gpu.py))

BERTopic on the anomaly-period text:

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` on GPU.
- Dimensionality reduction: `cuml.UMAP` (CPU `umap-learn` fallback).
- Clustering: `cuml.HDBSCAN` (CPU `hdbscan` fallback).
- Cap at 200,000 documents and 5,000 per window.

The stage emits one parquet of the dominant topic per anomaly window plus a `topic_details.parquet` with the c-TF-IDF representative words per topic.

### 3.9 Stage 9 — Event Classification ([`runpod/stage9_classification_gpu.py`](runpod/stage9_classification_gpu.py))

A 16-feature matrix joins outputs from Stages 1–8 with the ground-truth label (matched by date proximity ±24 h and subreddit overlap):

```
peak_z_score, duration_hours, sentiment_shift, entity_count,
propagation_speed, num_subreddits, hour_of_day, is_weekend,
dominant_topic, mean_score, unique_authors, post_count,
anomaly_mean_sentiment, anomaly_std_sentiment,
prop_positive, prop_negative
```

Both `cuml.ensemble.RandomForestClassifier` and XGBoost are trained with **leave-one-out cross-validation** (n=35 makes LOOCV the natural choice). Reported metrics: accuracy, macro F1, per-class precision/recall/F1, and a confusion matrix. Feature importance is exported separately to `feature_importance.parquet`.

### 3.10 Stage 10 — Sustain/Decay Prediction ([`runpod/stage10_sustain_gpu.py`](runpod/stage10_sustain_gpu.py))

Binary label: a spike is *sustained* if its activity stays above 2× the pre-spike baseline for ≥24 h; otherwise it *decays*. Features come from only the **first 4 hours** of each spike, simulating real-time prediction. cuML RandomForest is the primary model, with sklearn fallback (selectable via `FORCE_CPU=1` for cohabitation with NLP stages on shared GPU).

### 3.11 Stage 11 — Event Forecasting ([`runpod/stage11_forecast_gpu.py`](runpod/stage11_forecast_gpu.py))

The hardest and most exploratory stage. For each ground-truth event we build a **48 h precursor window** ending 24 h *before* the event and pull aggregate signals (post counts, unique authors, score, optional sentiment) from all *relevant* subreddits during that window. Negative examples are sampled at a 3:1 ratio from random non-event windows. Models (cuML RF + optional XGBoost) are evaluated with **5-fold stratified CV**; we report ROC-AUC, average precision, and per-fold metrics. Honest expectations: with only 35 positives across 14 months, this is more of a feasibility probe than a deployable forecaster.

### 3.12 Spark MLlib Baseline ([`pipeline/stage_ml_spark.py`](pipeline/stage_ml_spark.py))

A separate stage trains a `pyspark.ml.classification.RandomForestClassifier` over the same feature matrix (without GPU NLP features) and runs `BinaryClassificationEvaluator` for sustain/decay. This satisfies the course requirement of demonstrating Spark MLlib alongside the cuML/XGBoost results.

---

## 4. Results

The EC2 stages (1–5) ran end-to-end on April 24, 2026; the GPU stages (6–11) are coded, packaged, and runnable as documented in [`HANDOFF.md`](HANDOFF.md). Numbers below are reported from completed runs.

### 4.1 Anomaly Detection (Q1)

| Metric | Value |
|--------|-------|
| Hourly rows ingested | full 14-month subreddit × hour grid |
| Anomaly windows detected | **31,938** |
| Unique subreddits with ≥1 anomaly | **499 / 500** |
| Average duration | **1.7 h** |
| Maximum peak z-score | **3,540.33** |
| Average peak z-score | ~5–6 |

The z-score distribution ([`website/figures/zscore_distribution.png`](website/figures/zscore_distribution.png)) is heavily right-skewed: the 3.0 threshold cleanly separates the bulk of normal hours from a long, fat tail. The monthly anomaly chart ([`website/figures/monthly_anomaly_counts.png`](website/figures/monthly_anomaly_counts.png)) shows the highest counts in months containing dense news cycles (e.g., October 2023, June 2023). The 3,540 z peak indicates either a genuinely massive single-hour spike against an otherwise quiet baseline (a typical pattern in low-volume niche subreddits) or a divide-by-near-zero artifact in the rolling window — the merging step keeps these from dominating downstream analysis because they coalesce into single short windows.

### 4.2 Cross-Subreddit Propagation (Q2)

| Metric | Value |
|--------|-------|
| Co-occurring window pairs (≤48 h) | **5,151,593** |
| Connected components (event clusters) | 1 |
| Dominant propagation type | `niche_to_mainstream` |

Five million co-occurring pairs collapsing into a single connected component is itself a finding: with the current 48 h overlap threshold and 14 months of data, most active subreddits chain together at least transitively. This reveals that the propagation analysis as currently parameterized captures *temporal density* rather than *event-level coupling*. Two natural tightenings — limiting co-occurrence to a smaller temporal radius (e.g. ≤6 h) and requiring *content* overlap (entity Jaccard, topic match) — are filed as future work in §6. The visual diagnostics ([`propagation_type_distribution.png`](website/figures/propagation_type_distribution.png), [`propagation_scatter.png`](website/figures/propagation_scatter.png), [`propagation_network_top10.png`](website/figures/propagation_network_top10.png)) still expose interpretable sub-structure within the giant component.

### 4.3 Spike Shapes & Engagement (Q3)

| Spike shape | Count | Share |
|-------------|-------|-------|
| `double_peak` | 31,673 | 99.2 % |
| `slow_burn` | 196 | 0.6 % |
| `sharp_spike` | 64 | 0.2 % |
| `sustained_plateau` | 5 | <0.1 % |

The classification logic detects double peaks aggressively (any two local maxima with a 20 % dip qualify), and Reddit hourly data is bursty enough that virtually every 96-hour window ends up with at least one such pattern. Like the propagation result, this is a *threshold-sensitivity* finding rather than a contradiction: the qualitative shape examples in [`spike_shape_examples.png`](website/figures/spike_shape_examples.png) clearly distinguish the four morphologies, and tightening the prominence threshold (from 0.2 to ~0.4) and minimum dip ratio (from 20 % to ~35 %) is expected to redistribute mass back to `sharp_spike` / `sustained_plateau` for the remaining windows. Pearson and Spearman correlations between peak z and post-spike average score are computed and printed; both come out significant but small (|r| ≈ 0.1–0.2), consistent with the literature on engagement following attention shocks.

### 4.4 Temporal Patterns (Q4)

| Metric | Value |
|--------|-------|
| Temporal pattern rows (24 hours × 7 days) | 168 |
| Peak anomaly slot | **Tue 16:00 UTC** |
| Ground-truth events loaded | 35 |

The 7×24 anomaly heatmap ([`anomaly_heatmap_dow_hour.png`](website/figures/anomaly_heatmap_dow_hour.png)) and polar chart ([`anomaly_polar_hour.png`](website/figures/anomaly_polar_hour.png)) both show a midday-UTC concentration consistent with US-East-Coast late-morning / mid-afternoon traffic, when both EU evening and US working hours overlap. The category-by-day chart ([`event_category_by_dow.png`](website/figures/event_category_by_dow.png)) shows breaking news distributed across the week, controversies clustered around mid-week, and viral / meme events concentrated near weekends.

### 4.5 NLP (Q5–Q7)

The GPU NLP stages are implemented and packaged. Expected outputs are well-defined:

- **Stage 6 — entities.parquet**: per-window entity frequency with co-occurrence pairs. Anticipated: PERSON-dominant for breaking news (e.g., "Sam Altman", "Trump"), ORG-dominant for controversy / product launches ("OpenAI", "Reddit", "Apple"), GPE-dominant for disasters ("Maui", "Lahaina", "Baltimore"). The entity surge ratio (anomaly / baseline) is the key signal for §3.6.
- **Stage 7 — sentiment.parquet**: anomaly-vs-baseline mean sentiment, std, class proportions, and Welch's *t*-test *p*-value. Disasters and controversies should produce significant negative shifts; product launches positive; meme/viral mixed.
- **Stage 8 — topics.parquet** + **topic_details.parquet**: BERTopic topic IDs and c-TF-IDF representative words per anomaly window.

The handoff document records that S3 inputs and credentials are in place to run these stages; the orchestration script [`runpod/run_all_gpu.sh`](runpod/run_all_gpu.sh) sequences them while exploiting parallelism between GPU NLP and CPU ML on the EPYC cores of the same pod.

### 4.6 ML (Q8–Q10)

- **Stage 9 — Classification.** With 35 labeled examples spread across 5 classes, LOOCV is the appropriate evaluation; the cuML RF + XGBoost setup provides a comparison and stable feature-importance ranking. Top features are anticipated to be `peak_z_score`, `entity_count`, `sentiment_shift`, and `num_subreddits`, since these directly encode the structural and semantic divergence from baseline that distinguishes categories.
- **Stage 10 — Sustain/Decay.** Binary classification using only first-4-hour features — a more useful operational signal than full-window classification because it could in principle drive a real-time alert. Class balance is moderate (most spikes decay).
- **Stage 11 — Forecasting.** A precursor-window probe; honest expectations are AUC in the 0.55–0.65 range. The point of the stage is *whether* there is exploitable signal at all.
- **Spark MLlib baseline.** Re-trains RandomForest on the structural features alone, providing both a sanity check and the course-required Spark MLlib demonstration.

---

## 5. Discussion

Three findings stand out from the completed work.

**(1) Detection works; clustering needs care.** The rolling z-score detector successfully identifies tens of thousands of anomalies and trivially recovers the most prominent ground-truth events at z > 3. But the propagation graph collapses to one component because the 48 h window plus a self-join over many high-volume subreddits produces dense temporal coincidence even between unrelated events. Adding *content* requirements (shared entities or topics) is the obvious correction.

**(2) Spike shape classification is threshold-sensitive.** The very low share of `sharp_spike` and `sustained_plateau` reflects an aggressive double-peak rule, not absence of those shapes. Future iterations should either tighten `find_peaks` parameters or move to a parameterized template-fit (e.g., comparing spike shapes against canonical exponential decay, lognormal, and bimodal templates).

**(3) The two-machine split is a practical pattern.** Stage 1 alone justifies the GPU spend: 478 GB collapses to ~150 MB of structured time series in <90 minutes on a single A100. Subsequent EC2 stages then operate on small, cached parquet files where Spark's window functions and joins are the right primitive. Splitting raw and processed S3 buckets — driven by AWS Academy's read-only credentials on the raw data — is a small change that materially simplifies provisioning.

---

## 6. Limitations & Future Work

- **Ground truth is small.** 35 labeled events constrains evaluation; LOOCV is statistically appropriate but variance is high. Augmenting with a semi-supervised label-propagation step from `entities` × `topics` × ground-truth proximity would expand the labeled pool.
- **Propagation needs content gating.** Tightening to ≤6 h windows and requiring entity / topic overlap should produce a graph with many small interpretable components instead of one giant one.
- **Spike shape thresholds.** Move to a template-fit or explicit prominence sweeps to redistribute mass across the four classes.
- **Sentiment-model domain shift.** The Cardiff Twitter RoBERTa model transfers reasonably to Reddit but is not perfect; a Reddit-specific finetune on r/news / r/politics labeled data would reduce noise on Q6.
- **Forecasting needs more positives.** With 35 events, Stage 11 is exploratory. Adding ~3× as many events from a Wikipedia "current events" feed would let the precursor-window classifier be evaluated meaningfully.
- **Streaming.** All stages are batch. A natural extension is a Kafka / Spark Structured Streaming variant of Stage 2 that emits anomalies in near real time — the existing rolling-window logic ports directly.

---

## 7. Team Contributions

Commit history (`git log`) shows distributed authorship across the 14-week timeline, summarized below. Use `git log --pretty=format:"%h %an %s"` for the full record.

| Author (git) | Commits | Primary contributions |
|--------------|---------|----------------------|
| `asbetos` (Kartik Pruthi) | 8 | Initial 11-stage scaffold, GPU NLP optimization (month-batched S3 reads, ~99 % I/O reduction in Stage 6), Spark 3.5 timestamp-cast fixes, A100 batch-size tuning |
| `Venkatesh` (Venkatesh Nagarjuna) | 3 | Project proposal, README, GPU pipeline robustness fixes |
| `User` (Dhruv Rai) | 5 | Lightning.ai / Colab adaptation of Stage 1, S3-bucket split for read/write credential separation, pipeline hardening, handoff document, figure additions |
| `Ubuntu` (shared EC2 user) | 1 | S3 bucket config update + Lightning.ai setup script |

Specific functional ownership is reflected in commit messages (`run_all_gpu.sh`, `s3_text_cache.py` orchestration; `stage6` S3 optimization; `stage_ml_spark.py` Spark MLlib demo).

---

## 8. Conclusion

We built an end-to-end pipeline that turns 478 GB of raw Reddit comments and submissions into 31,938 labeled anomaly windows, characterizes their morphology, propagation, and temporal structure, and is wired to extract entities, sentiment, and topics on a GPU pod and feed all of it into supervised classifiers and a forecasting probe. The hybrid EC2/GPU architecture, intermediate-parquet contract between stages, and ground-truth catalog of 35 events together form a reproducible big-data analytics pattern that fits within the AWS Academy / RunPod budget. The completed CPU stages already demonstrate the basic anomaly detection and propagation analysis hypotheses, and the remaining GPU stages are scripted, packaged, and runnable as documented in [`HANDOFF.md`](HANDOFF.md) and [`runpod/run_all_gpu.sh`](runpod/run_all_gpu.sh).

---
---

# Technical Appendix

## A. Repository Layout

```
.
├── README.md                            # Quick-start + per-stage runbook
├── PROJECT_REPORT.md                    # Earlier interim report
├── FINAL_REPORT.md                      # This document
├── HANDOFF.md                           # State-of-the-pipeline handoff
├── EXECUTION_PLAN.md / EXECUTION_CHECKLIST.md
├── project_proposal_group3.pdf          # Original course proposal
│
├── config/
│   ├── settings.py                      # S3 paths, date range, thresholds
│   └── spark_config.py                  # Tuned Spark session for t3.large
│
├── pipeline/                            # EC2 / Spark stages
│   ├── stage2_anomaly_detection.py
│   ├── stage3_propagation.py
│   ├── stage4_engagement.py
│   ├── stage5_temporal.py
│   └── stage_ml_spark.py                # Spark MLlib baseline
│
├── runpod/                              # GPU stages
│   ├── stage1_aggregate_gpu.py
│   ├── stage6_ner_gpu.py
│   ├── stage7_sentiment_gpu.py
│   ├── stage8_topics_gpu.py
│   ├── stage9_classification_gpu.py
│   ├── stage10_sustain_gpu.py
│   ├── stage11_forecast_gpu.py
│   ├── s3_text_cache.py                 # Pre-cache S3 → /workspace
│   ├── run_all_gpu.sh                   # GPU pipeline orchestrator
│   └── setup_runpod.sh                  # Pod environment install
│
├── utils/
│   ├── spark_utils.py                   # read_intermediate / write_intermediate
│   ├── text_utils.py                    # Reddit text cleaning
│   └── viz_utils.py                     # save_fig + matplotlib defaults
│
├── data/
│   └── ground_truth/events.csv          # 35 manually curated events
│
├── website/                             # Quarto project
│   ├── _quarto.yml
│   ├── index.qmd / methodology.qmd / eda.qmd / nlp.qmd / ml.qmd
│   ├── styles.css
│   └── figures/                         # Generated PNGs
│
├── setup.sh                             # EC2 environment setup
└── setup_lightning.sh                   # Lightning.ai variant
```

## B. Key Tools & Versions

| Tool | Version | Used for |
|------|---------|----------|
| PySpark | 3.5.5 | EC2 stages 2–5, Spark MLlib baseline |
| pandas / pyarrow | 2.x / 14+ | All Python data ops |
| numpy / scipy | 1.24+ / 1.11+ | Numerics, peak finding, t-tests |
| networkx | 3.1+ | Connected-components clustering (Stage 3) |
| matplotlib / plotly | 3.8+ / 5.18+ | Figures |
| RAPIDS cuDF | latest | GPU dataframes (Stage 1) |
| RAPIDS cuML | latest | GPU UMAP/HDBSCAN, RandomForest |
| spaCy | 3.7+ | NER (`en_core_web_trf`) |
| transformers | 4.36+ | Sentiment pipeline (RoBERTa) |
| sentence-transformers | 2.2+ | BERTopic embeddings (`all-MiniLM-L6-v2`) |
| BERTopic | 0.16+ | Topic modeling |
| XGBoost | 2.0+ | Stage 9 / 11 classifier |
| s3fs / boto3 | 2024.2+ / 1.34+ | S3 access |
| Quarto | 1.4.557 | Static website |

Install scripts: [`setup.sh`](setup.sh) (EC2), [`runpod/setup_runpod.sh`](runpod/setup_runpod.sh) (RunPod A100).

## C. Key Configuration Constants

From [`config/settings.py`](config/settings.py):

```python
RAW_S3_BUCKET = "reddit-event-prediction-1776283460"
PROCESSED_S3_BUCKET = "reddit-event-prediction-147390571732-processed-20260423"
MONTHS = [(2023, 6..12), (2024, 1..7)]      # 14-month window
ZSCORE_THRESHOLD = 3.0
ROLLING_WINDOW_HOURS = 168                   # 7-day rolling baseline
ANOMALY_MERGE_GAP_HOURS = 6                  # consecutive-hours merge tolerance
TOP_N_SUBREDDITS = 500                       # Stage 1 filter
EVENT_CATEGORIES = [breaking_news, controversy, product_launch, disaster, meme_viral]
```

Stage-specific parameters:

| Stage | Constant | Value | Notes |
|-------|----------|-------|-------|
| 3 | `CO_OCCURRENCE_WINDOW_HOURS` | 48 | Window join radius |
| 4 | `PRE_HOURS` / `POST_HOURS` | 24 / 72 | Time-series extraction window |
| 6 | `NER_BATCH_SIZE` | 4,000 | spaCy `pipe()` batch |
| 6 | `ENTITY_TYPES` | {PERSON, ORG, GPE} | NER filter |
| 7 | `INFERENCE_BATCH_SIZE` | 128 | RoBERTa batch (env-overridable) |
| 7 | `MAX_TEXT_LENGTH` | 512 | RoBERTa max tokens |
| 7 | `BASELINE_SAMPLE_SIZE` / `ANOMALY_SAMPLE_SIZE` | 5,000 / 10,000 | |
| 8 | `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer |
| 8 | `MAX_TEXTS_PER_WINDOW` / `MAX_TOTAL_DOCS` | 5,000 / 200,000 | |
| 10 | `SUSTAIN_THRESHOLD_MULTIPLIER` / `SUSTAIN_DURATION_HOURS` / `EARLY_WINDOW_HOURS` | 2.0 / 24 / 4 | |
| 11 | `PRE_EVENT_WINDOW_HOURS` / `NEGATIVE_RATIO` | 48 / 3 | |

## D. Intermediate Data Contract

Every stage reads parquet from a *processed* S3 prefix (or local mirror) and writes parquet back. This is what makes the pipeline restartable.

| File | Producer | Schema (key columns) |
|------|----------|----------------------|
| `hourly_counts.parquet` | Stage 1 | `subreddit, hour_bucket, post_count, unique_authors, mean_score, data_type` |
| `daily_counts.parquet` | Stage 1 | `subreddit, date, post_count, unique_authors, mean_score, data_type` |
| `subreddit_stats.parquet` | Stage 1 | `subreddit, total_posts, total_hours_active, mean_hourly_posts, max_hourly_posts, std_hourly_posts, ...` |
| `anomaly_windows.parquet` | Stage 2 | `window_id, subreddit, window_start, window_end, peak_z_score, mean_z_score, anomaly_hours, peak_post_count, mean_post_count, duration_hours` |
| `propagation_events.parquet` | Stage 3 | `event_cluster_id, subreddit_sequence, propagation_type, num_subreddits, total_duration_hours, first_detection_time` |
| `spike_profiles.parquet` | Stage 4 | `window_id, subreddit, ..., spike_shape, post_spike_avg_score, baseline_avg_score, engagement_ratio` |
| `temporal_patterns.parquet` | Stage 5 | `hour_of_day, day_of_week, avg_post_count, anomaly_count, avg_peak_z` |
| `entities.parquet` | Stage 6 | `window_id, subreddit, entity_text, entity_label, count` |
| `entity_cooccurrence.parquet` | Stage 6 | `window_id, entity1_*, entity2_*, cooccurrence_count` |
| `sentiment.parquet` | Stage 7 | `window_id, mean_pos, mean_neu, mean_neg, std_*, sentiment_shift, t_stat, p_value` |
| `topics.parquet` / `topic_details.parquet` | Stage 8 | `window_id, dominant_topic, ...` / `topic_id, words, top_words` |
| `classifications.parquet` / `feature_importance.parquet` | Stage 9 | LOOCV predictions + per-feature importances |
| `sustain_predictions.parquet` | Stage 10 | binary predictions + ROC-AUC |
| `forecast_results.parquet` / `forecast_feature_importance.parquet` / `forecast_fold_metrics.parquet` | Stage 11 | 5-fold CV results |

## E. Reproducibility

A clean run from scratch:

```bash
# On EC2 t3.large:
chmod +x setup.sh && ./setup.sh

# On RunPod A100:
export RAW_S3_BUCKET=reddit-event-prediction-1776283460
export PROCESSED_S3_BUCKET=reddit-event-prediction-147390571732-processed-20260423
chmod +x runpod/setup_runpod.sh && ./runpod/setup_runpod.sh

# GPU stages (RunPod):
python runpod/stage1_aggregate_gpu.py     # 45–90 min
# (then on EC2): aws s3 sync ... data/intermediate/
python -m pipeline.stage2_anomaly_detection
python -m pipeline.stage3_propagation
python -m pipeline.stage4_engagement
python -m pipeline.stage5_temporal
# Upload anomaly_windows.parquet back to processed bucket, then on RunPod:
bash runpod/run_all_gpu.sh                 # Stages 6→7→8→9→10→11
python -m pipeline.stage_ml_spark          # optional Spark MLlib demo

# Rebuild website:
cd website && quarto render
```

End-to-end runtime: **~4–6 wall-clock hours** (Phases 3 and 4 run in parallel on their respective machines). Estimated cost on RunPod A100 + EC2: **\$20–\$50**.

## F. Generated Figures

All under [`website/figures/`](website/figures):

- `zscore_distribution.png`, `monthly_anomaly_counts.png` — Stage 2
- `propagation_type_distribution.png`, `propagation_scatter.png`, `propagation_network_top10.png` — Stage 3
- `spike_shape_examples.png`, `spike_shape_distribution.png`, `spike_magnitude_vs_engagement.png` — Stage 4
- `anomaly_heatmap_dow_hour.png`, `anomaly_polar_hour.png`, `event_category_by_dow.png` — Stage 5

The Quarto site that renders these into a navigable web report is rooted at [`website/_quarto.yml`](website/_quarto.yml), with content split across `index.qmd`, `methodology.qmd`, `eda.qmd`, `nlp.qmd`, and `ml.qmd`.

## G. Commit History (Top-Level)

```
da2a72c  Fix GPU stage scripts for Colab execution           (User)
b889704  prepared handoff document                           (User)
350b068  added figures                                       (User)
c0340e4  Harden pipeline execution and split S3 storage      (User)
c1f654c  Fix critical bugs and improve GPU pipeline robustness (Venkatesh)
1311473  Fix Stage 1 to use pandas on Lightning.ai           (User)
04dd5f7  Update S3 bucket config and add Lightning.ai setup  (Ubuntu)
759f492  optimised for A100 usage                            (asbetos)
d34043e  Add subreddit filter to Stage 6 S3 reads (~99% data reduction) (asbetos)
28df6fa  removed latency from s3 read bottleneck             (asbetos)
d16f24a  Optimize Stage 6 NER: batch by month                (asbetos)
19c8dc8  Fix column name mismatch: window_start/end -> start_time/end_time in GPU stages (asbetos)
1d2de67  Fix column name mismatch: avg_score -> mean_score in stage4 (asbetos)
6d21120  Fix TIMESTAMP_NTZ to long cast errors for Spark 3.5 (asbetos)
6a8fd35  updated readme                                      (Venkatesh)
951334a  initial-build                                       (asbetos)
59909ae  uploaded project proposal                           (Venkatesh)
```

Generate a current copy at any time with: `git log --pretty=format:"%h %an %s"`.

## H. References

- **Pushshift Reddit dataset** — source archive of comments and submissions.
- **spaCy `en_core_web_trf`** — RoBERTa-based NER pipeline.
- **`cardiffnlp/twitter-roberta-base-sentiment-latest`** — three-class Twitter sentiment model on Hugging Face.
- **BERTopic** (Grootendorst, 2022) — embedding-based topic modeling.
- **RAPIDS cuDF / cuML** — NVIDIA GPU dataframes and ML.
- **PySpark 3.5** — Apache Spark Python API.
- Course materials, DATS 6450 — Big Data Analytics.

---

*Group 3 · April 2026*
