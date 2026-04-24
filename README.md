# Event Prediction Using Reddit Data

An early warning system that detects and predicts real-world events by analyzing activity patterns across 500 subreddits. Processes 478 GB of Reddit data (June 2023 -- July 2024) through an 11-stage pipeline spanning anomaly detection, cross-subreddit propagation analysis, NLP (NER, sentiment, topic modeling), event classification, and forecasting.

**Course:** DATS 6450 | **Team:** Group 3

## Architecture

The pipeline runs across two compute environments:

| Environment | Spec | Stages |
|---|---|---|
| **EC2** (t3.large) | 2 vCPU, 8 GB RAM, PySpark | Stages 2--5, Spark ML |
| **RunPod** (A100 80GB) | GPU, RAPIDS/cuML | Stages 1, 6--11 |

Raw input is read from `s3://reddit-event-prediction-1776283460/reddit-data/parquet/`.

Processed intermediate data can be written to a separate S3 bucket, for example `s3://reddit-event-prediction-147390571732-processed-20260423/reddit-data/intermediate/`.

## Project Structure

```
├── config/                  # Configuration and constants
│   ├── settings.py          #   S3 paths, date ranges, thresholds
│   └── spark_config.py      #   PySpark session factory (EC2)
├── pipeline/                # EC2 Spark stages (CPU)
│   ├── stage2_anomaly_detection.py
│   ├── stage3_propagation.py
│   ├── stage4_engagement.py
│   ├── stage5_temporal.py
│   └── stage_ml_spark.py    #   Spark MLlib demo (Q8/Q9)
├── runpod/                  # RunPod GPU stages
│   ├── stage1_aggregate_gpu.py
│   ├── stage6_ner_gpu.py
│   ├── stage7_sentiment_gpu.py
│   ├── stage8_topics_gpu.py
│   ├── stage9_classification_gpu.py
│   ├── stage10_sustain_gpu.py
│   ├── stage11_forecast_gpu.py
│   └── setup_runpod.sh      #   RunPod environment setup
├── utils/                   # Shared utilities
│   ├── spark_utils.py       #   S3/local parquet I/O helpers
│   ├── text_utils.py        #   Reddit text cleaning
│   └── viz_utils.py         #   Matplotlib/Plotly helpers
├── data/
│   └── ground_truth/
│       └── events.csv       #   35 manually curated events
├── website/                 # Quarto project website
├── setup.sh                 # EC2 environment setup
└── project_proposal_group3.pdf
```

## Prerequisites

- **AWS credentials** configured with read access to the raw input bucket `reddit-event-prediction-1776283460`
- **AWS credentials** configured with write access to the processed output bucket `reddit-event-prediction-147390571732-processed-20260423`
- **EC2 instance** (t3.large or larger) running Ubuntu with Java 17
- **RunPod pod** with an A100 80GB GPU, CUDA drivers, and a PyTorch/CUDA base image
- Raw Reddit dataset already uploaded to `s3://reddit-event-prediction-1776283460/reddit-data/parquet/`

## Step-by-Step Execution Guide

### Phase 0: Environment Setup

Run these on their respective machines. Only needed once.

**On EC2:**
```bash
chmod +x setup.sh
./setup.sh
```
Installs PySpark, Python packages, Quarto, and system fonts.
> Estimated time: ~5 minutes

**On RunPod:**
```bash
export RAW_S3_BUCKET="reddit-event-prediction-1776283460"
export PROCESSED_S3_BUCKET="reddit-event-prediction-147390571732-processed-20260423"

chmod +x runpod/setup_runpod.sh
./runpod/setup_runpod.sh
```
Installs RAPIDS (cuDF/cuML), transformers, spaCy with GPU, BERTopic, and configures AWS credentials.
> Estimated time: ~10--15 minutes

RunPod image guidance:
- Use a `PyTorch/CUDA` template image, not a plain Python image.
- The repo does not pin a specific PyTorch version; it expects `torch` to already be installed and CUDA-enabled in the base image.
- Verify with `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
- Safest choice: Python `3.10` or `3.11`, since RAPIDS compatibility is usually smoother than `3.12`.

---

### Phase 1: Data Aggregation (RunPod GPU)

**Stage 1** -- Aggregate raw Reddit data into hourly/daily counts.

```bash
export RAW_S3_BUCKET="reddit-event-prediction-1776283460"
export PROCESSED_S3_BUCKET="reddit-event-prediction-147390571732-processed-20260423"

python runpod/stage1_aggregate_gpu.py
```

Reads 478 GB of raw comments and submissions from S3 month-by-month using cuDF on GPU. Produces hourly counts, daily counts, and per-subreddit statistics. Supports checkpointing -- if interrupted, rerun and it will resume from the last completed month.

- **Input:** Raw Reddit parquet on S3 (478 GB)
- **Output:** `hourly_counts.parquet`, `daily_counts.parquet`, `subreddit_stats.parquet` on S3
- **Estimated time: 45--90 minutes**

After completion, download outputs to EC2:
```bash
# On EC2
aws s3 sync s3://reddit-event-prediction-147390571732-processed-20260423/reddit-data/intermediate/ data/intermediate/ \
  --exclude "*" \
  --include "hourly_counts.parquet/*" \
  --include "daily_counts.parquet/*" \
  --include "subreddit_stats.parquet/*"
```

---

### Phase 2: Anomaly Detection (EC2 Spark)

**Stage 2** -- Detect anomalous activity spikes using z-score analysis.

```bash
python -m pipeline.stage2_anomaly_detection
```

Computes 7-day rolling statistics per subreddit, flags hours with z > 3.0, and merges consecutive anomalous hours into contiguous windows.

- **Input:** `hourly_counts.parquet`
- **Output:** `anomaly_windows.parquet`, figures in `website/figures/`
- **Estimated time: 5--10 minutes**

After completion, upload to S3 for GPU stages:
```bash
aws s3 cp data/intermediate/anomaly_windows.parquet \
  s3://reddit-event-prediction-147390571732-processed-20260423/reddit-data/intermediate/anomaly_windows.parquet --recursive
```

---

### Phase 3: EC2 Stages (can run in parallel)

These stages all depend on Stage 2 output but are independent of each other.

**Stage 3** -- Cross-subreddit propagation analysis:
```bash
python -m pipeline.stage3_propagation
```
Finds co-occurring anomalies across subreddits within 48h, builds event clusters via graph analysis, classifies propagation type (simultaneous, niche-to-mainstream, top-down).

- **Input:** `anomaly_windows.parquet`, `hourly_counts.parquet`
- **Output:** `propagation_events.parquet`, figures
- **Estimated time: 5--10 minutes**

**Stage 4** -- Spike shape & engagement analysis:
```bash
python -m pipeline.stage4_engagement
```
Classifies spike shapes (sharp spike, sustained plateau, double peak, slow burn) and correlates magnitude with post-spike engagement.

- **Input:** `anomaly_windows.parquet`, `hourly_counts.parquet`
- **Output:** `spike_profiles.parquet`, figures
- **Estimated time: 10--15 minutes**

**Stage 5** -- Temporal pattern analysis:
```bash
python -m pipeline.stage5_temporal
```
Analyzes hour-of-day and day-of-week distributions, maps event categories to temporal patterns using ground truth.

- **Input:** `anomaly_windows.parquet`, `hourly_counts.parquet`, `data/ground_truth/events.csv`
- **Output:** `temporal_patterns.parquet`, figures
- **Estimated time: 5--10 minutes**

---

### Phase 4: GPU NLP Stages (RunPod, can run in parallel)

These stages all depend on `anomaly_windows.parquet` being in S3 but are independent of each other.

**Stage 6** -- Named Entity Recognition:
```bash
python runpod/stage6_ner_gpu.py
```
Runs spaCy transformer NER (`en_core_web_trf`) on text from each anomaly window. Extracts PERSON, ORG, GPE entities and co-occurrence pairs.

- **Input:** `anomaly_windows.parquet`, raw Reddit text from S3
- **Output:** `entities.parquet`, `entity_cooccurrence.parquet`
- **Estimated time: 2--4 hours**

**Stage 7** -- Sentiment Analysis:
```bash
python runpod/stage7_sentiment_gpu.py
```
Compares sentiment during vs. before anomaly windows using `twitter-roberta-base-sentiment-latest`. Computes sentiment shift with statistical significance testing.

- **Input:** `anomaly_windows.parquet`, raw Reddit text from S3
- **Output:** `sentiment.parquet`
- **Estimated time: 2--4 hours**

**Stage 8** -- Topic Modeling:
```bash
python runpod/stage8_topics_gpu.py
```
BERTopic with GPU-accelerated UMAP/HDBSCAN. Embeds up to 200k documents, clusters topics, assigns dominant topic per anomaly window.

- **Input:** `anomaly_windows.parquet`, raw Reddit text from S3
- **Output:** `topics.parquet`, `topic_details.parquet`
- **Estimated time: 30--90 minutes**

---

### Phase 5: Classification & Prediction (RunPod GPU)

These stages require outputs from Phase 4.

**Stage 9** -- Event Classification (requires Stages 6 + 7 + 8):
```bash
python runpod/stage9_classification_gpu.py
```
5-class event classification (breaking news, controversy, product launch, disaster, meme/viral) using cuML RandomForest and XGBoost with LOOCV.

- **Input:** `anomaly_windows`, `entities`, `sentiment`, `topics`, `ground_truth/events.csv`
- **Output:** `classifications.parquet`, `feature_importance.parquet`
- **Estimated time: 10--20 minutes**

**Stage 10** -- Sustain/Decay Prediction (requires Stages 1 + 2, optionally 7):
```bash
python runpod/stage10_sustain_gpu.py
```
Predicts whether a spike will sustain (>2x baseline for >24h) or decay, using only features from the first 4 hours of each spike.

- **Input:** `hourly_counts`, `anomaly_windows`, `sentiment` (optional)
- **Output:** `sustain_predictions.parquet`, `sustain_feature_importance.parquet`
- **Estimated time: 10--20 minutes**

**Stage 11** -- Event Forecasting (requires Stage 1, optionally 7):
```bash
python runpod/stage11_forecast_gpu.py
```
Predicts whether a major event will occur in the next 24h based on precursor signals from a 48h lookback window. Uses 5-fold stratified cross-validation.

- **Input:** `hourly_counts`, `ground_truth/events.csv`, `sentiment` (optional)
- **Output:** `forecast_results.parquet`, `forecast_feature_importance.parquet`
- **Estimated time: 10--20 minutes**

---

### Phase 6: Spark ML Baseline (EC2, optional)

**Spark MLlib Demo** -- Course requirement demonstrating PySpark ML:
```bash
python -m pipeline.stage_ml_spark
```
Trains RandomForestClassifier for event classification and sustain/decay prediction using Spark MLlib. Prints metrics to console.

- **Input:** `anomaly_windows`, `spike_profiles`, `temporal_patterns`, `ground_truth/events.csv`
- **Output:** Console output only
- **Estimated time: 5--10 minutes**

---

### Phase 7: Website (EC2)

Build the Quarto project website:
```bash
cd website && quarto render
```

- **Estimated time: ~1 minute**

---

## Total Estimated Runtime

| Phase | Stages | Time |
|---|---|---|
| Setup | Environment setup | ~15 min |
| Phase 1 | Stage 1 (GPU aggregation) | 45--90 min |
| Phase 2 | Stage 2 (anomaly detection) | 5--10 min |
| Phase 3 | Stages 3--5 (parallel on EC2) | 10--15 min |
| Phase 4 | Stages 6--8 (parallel on RunPod) | 2--4 hours |
| Phase 5 | Stages 9--11 (sequential on RunPod) | 30--60 min |
| Phase 6 | Spark ML (optional) | 5--10 min |
| **Total** | | **~4--6 hours** |

Note: Phases 3 and 4 can run concurrently on their respective machines, reducing wall-clock time. The bottleneck is Phase 4 (NLP stages reading raw text from S3).

## Ground Truth Events

The pipeline validates against 35 manually curated events across 5 categories:

| Category | Count | Examples |
|---|---|---|
| Breaking News | 15 | Reddit API protest, Sam Altman firing, Trump assassination attempt |
| Controversy | 7 | Unity pricing backlash, Harvard president resignation |
| Product Launch | 7 | Threads launch, GPT-4 Turbo, Apple Vision Pro |
| Disaster | 4 | OceanGate Titan, Maui wildfires, Baltimore bridge collapse |
| Meme/Viral | 3 | Barbenheimer, Grimace shake trend |

See `data/ground_truth/events.csv` for the full list.
