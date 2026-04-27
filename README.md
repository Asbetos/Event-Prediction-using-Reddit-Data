# Reddit as an Early Warning System

Detect, characterize, and forecast real-world events from 478 GB of Reddit comments and submissions (June 2023 — July 2024) across the top 500 subreddits, using an 11-stage pipeline that splits the work between PySpark on EC2 and a RAPIDS / Hugging Face / cuML stack on a RunPod A100 GPU pod.

**Course:** DATS 6450 — Big Data Analytics · **Team:** Group 3 (Venkatesh Nagarjuna · Kartik Pruthi · Dhruv Rai)

> **The full final report (10-page narrative + technical appendix) is in [`FINAL_REPORT.md`](FINAL_REPORT.md). A condensed run sheet of the actual numerical results is in [`RESULTS.md`](RESULTS.md).**

## What's in this repo

```
.
├── FINAL_REPORT.md                  Final project report (narrative + technical appendix)
├── RESULTS.md                       One-page summary of the actual numbers
├── README.md                        This file
├── PROJECT_REPORT.md / HANDOFF.md   Earlier interim docs (kept for provenance)
├── EXECUTION_PLAN.md / EXECUTION_CHECKLIST.md
│
├── config/                          S3 paths, date range, thresholds, Spark session
├── pipeline/                        EC2 / Spark stages 2–5 + Spark MLlib baseline
├── runpod/                          GPU stages 1, 6–11 + orchestration scripts
├── utils/                           Shared spark / text / viz helpers
├── analysis/
│   ├── generate_eda_figures.py      Rebuild Stage 2–5 figures from local data
│   └── generate_report_figures.py   Build NLP/ML figures (Stages 6–11) from local data
│
├── data/
│   ├── ground_truth/events.csv      35 manually curated events
│   └── intermediate/                All 17 stage-output parquet files (~84 MB)
│
├── website/                         Quarto site (index, methodology, eda, nlp, ml)
│   └── figures/                     20 PNG figures referenced from FINAL_REPORT.md
│
├── setup.sh                         EC2 environment setup
├── setup_lightning.sh               Lightning.ai variant
└── project_proposal_group3.pdf
```

The intermediate parquet outputs from every stage ship with the repo under `data/intermediate/`, so the analysis and figure-generation steps work offline without re-running the cloud pipeline.

## Quick start (analysis only, offline)

```bash
conda activate base
cd Reddit-Event-Prediction-Final

# Rebuild every figure used in the report from local intermediate data
python analysis/generate_eda_figures.py
python analysis/generate_report_figures.py

# Optional: render the Quarto website
cd website && quarto render
```

Reads from `data/intermediate/*.parquet`, writes to `website/figures/*.png`. ~10 seconds total.

## Architecture (cloud, full pipeline)

| Environment | Spec | Stages | Why |
|---|---|---|---|
| **EC2** `t3.large` | 2 vCPU, 8 GB RAM, PySpark 3.5 | 2 – 5, Spark MLlib baseline | Distributed reads from S3, window functions, joins, aggregation |
| **RunPod / Lightning.ai** | A100 80 GB, RAPIDS, PyTorch | 1, 6 – 11 | cuDF aggregation of 478 GB; transformer NER/sentiment; BERTopic; cuML RandomForest / XGBoost |

S3 is the shared bus. We use a separate raw bucket (`reddit-event-prediction-1776283460`, read-only AWS Academy creds) and processed bucket (`reddit-event-prediction-147390571732-processed-20260423`, write creds) — see [`config/settings.py`](config/settings.py).

## End-to-end pipeline

| # | Stage | Where | Output | Runtime |
|---|---|---|---|---|
| 1 | GPU aggregation | RunPod | `hourly_counts`, `daily_counts`, `subreddit_stats` | 45 – 90 min |
| 2 | Anomaly detection (rolling z) | EC2 | `anomaly_windows` | 5 – 10 min |
| 3 | Cross-subreddit propagation | EC2 | `propagation_events` | 5 – 10 min |
| 4 | Spike shapes & engagement | EC2 | `spike_profiles` | 10 – 15 min |
| 5 | Temporal patterns | EC2 | `temporal_patterns` | 5 – 10 min |
| 6 | NER (spaCy `en_core_web_trf`) | RunPod | `entities`, `entity_cooccurrence` | 2 – 4 h |
| 7 | Sentiment (Cardiff Twitter RoBERTa) | RunPod | `sentiment` | 2 – 4 h |
| 8 | Topics (BERTopic + cuML UMAP/HDBSCAN) | RunPod | `topics`, `topic_details` | 30 – 90 min |
| 9 | Event classification (cuML RF + XGBoost, LOOCV) | RunPod | `classifications`, `feature_importance` | 10 – 20 min |
| 10 | Sustain / decay (cuML RF, first-4h features) | RunPod | `sustain_predictions`, `sustain_feature_importance` | 10 – 20 min |
| 11 | Event forecasting (5-fold stratified, 48 h precursor) | RunPod | `forecast_results`, `forecast_feature_importance`, `forecast_fold_metrics` | 10 – 20 min |
| — | Spark MLlib baseline (course req.) | EC2 | console metrics | 5 – 10 min |

End-to-end wall-clock: ~4 – 6 hours (Phases 3 and 4 can run in parallel on their respective machines). Cost on RunPod A100 + EC2 t3.large: ~\$20 – \$50.

## Headline results (full numbers in `RESULTS.md` and `FINAL_REPORT.md`)

| | |
|---|---|
| Comments processed | **1.19 B** |
| Submissions processed | **87.2 M** |
| Hourly time-series rows | **9.38 M** |
| Anomaly windows detected (z>3) | **31,938** across **499 / 500** subreddits |
| Stage 6 entities extracted | **63,614** rows / **36,082** unique entities / **1,738** windows |
| Stage 11 forecasting AUC | **0.955** (5-fold CV); F1 **0.77**; precision **0.85**; recall **0.70** |
| Stage 10 sustain AUC | **0.949**; class-imbalanced (493 decayed / 7 sustained), so positive precision/recall = 0 |

## Reproducing the full cloud pipeline

### 0. Environment setup

```bash
# On EC2
chmod +x setup.sh && ./setup.sh

# On RunPod
export RAW_S3_BUCKET=reddit-event-prediction-1776283460
export PROCESSED_S3_BUCKET=reddit-event-prediction-147390571732-processed-20260423
chmod +x runpod/setup_runpod.sh && ./runpod/setup_runpod.sh
```

### 1. Stage 1 — GPU aggregation (RunPod)

```bash
python runpod/stage1_aggregate_gpu.py
```

Streams 478 GB of raw parquet through cuDF month-by-month, with monthly checkpoints to S3 so an interrupted run resumes cleanly.

### 2. Stages 2 – 5 — EC2 / Spark

```bash
python -m pipeline.stage2_anomaly_detection
python -m pipeline.stage3_propagation
python -m pipeline.stage4_engagement
python -m pipeline.stage5_temporal
```

Stages 3 – 5 are independent; each only depends on Stage 2 + Stage 1 outputs.

### 3. Stages 6 – 11 — GPU NLP and ML (RunPod)

```bash
bash runpod/run_all_gpu.sh
```

Orchestrates: Stage 6 → 7 → 8 sequentially on the GPU, plus Stages 10 / 11 on CPU EPYC cores in parallel (`FORCE_CPU=1`), then Stage 9 last because it joins all earlier NLP outputs. Falls back gracefully when cuDF / cuML are unavailable.

### 4. Optional — Spark MLlib baseline + website

```bash
python -m pipeline.stage_ml_spark
cd website && quarto render
```

## Ground truth

35 manually curated events spanning June 2023 – July 2024:

| Category | n | Examples |
|---|---|---|
| Breaking news | 15 | Hamas attack, Sam Altman returns to OpenAI, Trump assassination attempt, total solar eclipse |
| Controversy | 7 | Reddit API protest, Sam Altman fired, Biden-Trump debate, Unity pricing |
| Product launch | 7 | Threads launch, iPhone 15, GPT-4o, Apple Intelligence |
| Disaster | 4 | OceanGate Titan, Maui wildfires, Baltimore bridge, Alaska Airlines blowout |
| Meme/viral | 3 | Barbenheimer, Taylor Swift Super Bowl, Black Friday |

Full list: [`data/ground_truth/events.csv`](data/ground_truth/events.csv).

## Tools & versions

- **PySpark** 3.5 · **RAPIDS** cuDF, cuML · **PyTorch** with CUDA · **Hugging Face transformers** 4.36+
- **spaCy** 3.7+ with `en_core_web_trf` · **`cardiffnlp/twitter-roberta-base-sentiment-latest`` · **BERTopic** + `all-MiniLM-L6-v2`
- **XGBoost** 2.0+ · **scikit-learn** 1.3+ · **NetworkX** 3.1+
- **Quarto** 1.4 · **matplotlib**, **plotly**, **wordcloud**

## Provenance

Commit history is preserved on the `main` branch (run `git log --pretty=format:"%h %an %s"`). Authors:

| Author | Commits | Primary contributions |
|---|---|---|
| `asbetos` (Kartik Pruthi) | 8 | Initial 11-stage scaffold; Stage 6 month-batched S3 reads (~99 % I/O reduction); Spark 3.5 timestamp casting; A100 batch tuning |
| `Venkatesh` | 3 | Project proposal; README; GPU robustness fixes |
| `User` (Dhruv Rai) | 5 | Lightning.ai / Colab adaptation; raw/processed S3 bucket split; pipeline hardening; handoff document; figure additions |
| `Ubuntu` | 1 | Bucket config + Lightning.ai setup |

## License & credits

Course project for DATS 6450, Spring 2026. Reddit data via the [Pushshift](https://pushshift.io) archive. NLP models from spaCy and Hugging Face Hub.
