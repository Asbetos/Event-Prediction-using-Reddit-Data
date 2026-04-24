# Event Prediction Using Reddit Data - Comprehensive Project Report

**Course:** DATS 6450 | **Team:** Group 3  
**Date:** April 2025

---

## Executive Summary

This project implements an **early warning system** that detects and predicts real-world events by analyzing activity patterns across **500 subreddits**. The system processes **478 GB of Reddit data** spanning from June 2023 to July 2024 through an **11-stage pipeline** that includes anomaly detection, cross-subreddit propagation analysis, NLP (NER, sentiment, topic modeling), event classification, and forecasting.

---

## 1. Project Overview

### 1.1 Objective
Develop a machine learning pipeline that:
- Detects anomalous activity spikes in subreddit communities
- Analyzes cross-subreddit propagation patterns
- Extracts entities, sentiments, and topics from anomalous content
- Classifies events into 5 categories (breaking news, controversy, product launch, disaster, meme/viral)
- Predicts spike sustain/decay and forecasts future events

### 1.2 Data Source
- **Dataset:** 478 GB of Reddit comments and submissions
- **Time Period:** June 2023 - July 2024 (14 months)
- **Subreddits:** Top 500 by activity volume
- **Format:** Parquet files partitioned by year/month

### 1.3 Ground Truth
- **35 manually curated events** across 5 categories:

| Category | Count | Examples |
|----------|-------|----------|
| Breaking News | 15 | Sam Altman firing, Trump assassination attempt, Reddit IPO |
| Controversy | 7 | Unity pricing backlash, Reddit API protest, Harvard president resignation |
| Product Launch | 7 | Threads launch, GPT-4 Turbo, Apple Vision Pro, GTA VI trailer |
| Disaster | 4 | OceanGate Titan, Maui wildfires, Baltimore bridge collapse |
| Meme/Viral | 3 | Barbenheimer, Grimace shake trend |

---

## 2. Tech Stack

### 2.1 Compute Infrastructure

| Environment | Specification | Stages |
|-------------|--------------|--------|
| **EC2 (t3.large)** | 2 vCPU, 8 GB RAM, PySpark | Stages 2-5 (Spark ML) |
| **RunPod/Lightning.ai** | A100 80GB GPU, RAPIDS/cuML | Stages 1, 6-11 |

### 2.2 Core Technologies

#### GPU-Accelerated Stack (RunPod)
| Component | Technology | Purpose |
|-----------|------------|---------|
| DataFrame Processing | cuDF / pandas | GPU-accelerated data loading |
| Machine Learning | cuML (RAPIDS) | GPU RandomForest, XGBoost |
| NLP - NER | spaCy + transformers | Named entity recognition |
| NLP - Sentiment | RoBERTa (cardiffnlp) | Twitter sentiment analysis |
| NLP - Topics | BERTopic + sentence-transformers | Topic modeling with UMAP/HDBSCAN |
| Dimensionality Reduction | cuML UMAP | GPU-accelerated embeddings |
| Clustering | cuML HDBSCAN | GPU-accelerated clustering |
| Storage Interface | s3fs | S3 data access |

#### Spark Stack (EC2)
| Component | Technology | Purpose |
|-----------|------------|---------|
| Distributed Processing | PySpark 3.5+ | Large-scale data processing |
| Window Functions | Spark SQL | Time-series analysis |
| Graph Analysis | NetworkX | Propagation network clustering |
| Visualization | Matplotlib | Statistical plots |

#### Data Storage
- **Primary Storage:** AWS S3 (s3://reddit-event-prediction-1776283460/)
- **Intermediate Data:** Parquet format on S3
- **Local Cache:** 100GB+ for temporary processing

---

## 3. Pipeline Architecture (11 Stages)

### Phase 1: Data Aggregation (GPU - RunPod)
**Stage 1: Aggregate 478GB Raw Reddit Data**
- Input: Raw parquet files (comments + submissions)
- Output: hourly_counts.parquet, daily_counts.parquet
- Time: 45-90 minutes

### Phase 2: Anomaly Detection (Spark - EC2)
**Stage 2: Detect Anomalous Activity Spikes**
- Method: 7-day rolling z-score > 3.0
- Output: anomaly_windows.parquet
- Time: 5-10 minutes

### Phase 3: Spark Analysis (Parallel on EC2)
- **Stage 3: Propagation Analysis (Q2)** - Cross-subreddit co-occurrence, graph clustering
- **Stage 4: Engagement Analysis (Q3)** - Spike shape classification, post-spike engagement
- **Stage 5: Temporal Analysis (Q4)** - Hour-of-day patterns, category-time mapping
- Time: 25-45 minutes total (can run in parallel)

### Phase 4: NLP Analysis (Parallel on RunPod GPU)
- **Stage 6: NER (Q5)** - spaCy en_core_web_trf on GPU
  - Output: entities.parquet, entity_cooccurrence.parquet
  - Time: 2-4 hours
- **Stage 7: Sentiment Analysis (Q6)** - RoBERTa twitter-roberta-base-sentiment-latest
  - Output: sentiment.parquet
  - Time: 2-4 hours
- **Stage 8: Topic Modeling (Q7)** - BERTopic with GPU UMAP/HDBSCAN
  - Output: topics.parquet, topic_details.parquet
  - Time: 30-90 minutes

### Phase 5: Classification and Prediction (RunPod GPU)
- **Stage 9: Event Classification (Q8)** - 5-class classification using cuML RF/XGB
  - Output: classifications.parquet, feature_importance.parquet
  - Time: 10-20 minutes
- **Stage 10: Sustain/Decay Prediction (Q9)** - Binary prediction using cuML RF
  - Output: sustain_predictions.parquet
  - Time: 10-20 minutes
- **Stage 11: Event Forecasting (Q10)** - Predict next 24h from 48h precursor window
  - Output: forecast_results.parquet
  - Time: 10-20 minutes

---

## 4. Data Preprocessing with GPUs

### 4.1 Stage 1: GPU-Accelerated Aggregation

**Challenge:** Processing 478 GB of raw Reddit data efficiently

**Solution:**
- **Month-by-month processing:** Reads data incrementally to manage memory
- **cuDF GPU DataFrames:** When available, uses GPU for aggregation operations
- **Column pruning:** Only reads necessary columns (subreddit, created_utc, author, score)
- **Checkpointing:** Saves monthly checkpoints to resume if interrupted

**Key Operations:**
```python
# Hourly aggregation per subreddit
hourly_counts = df.groupby(["subreddit", "hour_bucket"]).agg(
    post_count=("score", "count"),
    unique_authors=("author", "nunique"),
    mean_score=("score", "mean"),
)
```

**Output Files:**
- hourly_counts.parquet - Subreddit x hour activity metrics
- daily_counts.parquet - Aggregated daily metrics
- subreddit_stats.parquet - Per-subreddit summary statistics

### 4.2 Memory Management Strategy

```python
# GPU memory optimization
def process_one_month(year, month):
    # 1. Read only required columns
    df = pd.read_parquet(path, columns=["subreddit", "created_utc", "author", "score"])
    
    # 2. Aggregate immediately to reduce size
    hourly = aggregate_hourly(df)
    
    # 3. Delete raw data and garbage collect
    del df
    gc.collect()
    
    # 4. Clear GPU cache (safe method)
    torch.cuda.empty_cache()
```

### 4.3 NLP Stages GPU Processing (Stages 6-8)

**Stage 6: Named Entity Recognition**
- Model: spaCy en_core_web_trf (transformer-based)
- GPU acceleration via spacy.prefer_gpu()
- Batch processing with spacy.pipe()
- Entity types: PERSON, ORG, GPE
- Extracts entity co-occurrence pairs

**Stage 7: Sentiment Analysis**
- Model: cardiffnlp/twitter-roberta-base-sentiment-latest
- Runs on GPU via transformers pipeline
- Compares anomaly vs baseline sentiment
- Statistical significance testing (Welch's t-test)

**Stage 8: Topic Modeling**
- Framework: BERTopic
- Embeddings: sentence-transformers (all-MiniLM-L6-v2) on GPU
- Dimensionality Reduction: cuML UMAP (GPU) or CPU UMAP fallback
- Clustering: cuML HDBSCAN (GPU) or CPU HDBSCAN fallback
- Processes up to 200k documents

---

## 5. Current Project Status

### 5.1 Completed Tasks

| Component | Status | Details |
|-----------|--------|---------|
| Project Architecture | Complete | 11-stage pipeline fully designed |
| Code Implementation | Complete | All 11 stages implemented |
| Environment Setup Scripts | Complete | setup.sh (EC2), setup_runpod.sh (GPU) |
| Configuration | Complete | S3 paths, thresholds, constants |
| Utility Functions | Complete | S3 I/O, text cleaning, visualization |
| Ground Truth Dataset | Complete | 35 events across 5 categories |

### 5.2 Pipeline Stages Status

| Stage | Name | Platform | Status | Dependencies |
|-------|------|----------|--------|--------------|
| 1 | Data Aggregation | GPU | Ready | Raw data on S3 |
| 2 | Anomaly Detection | Spark | Ready | Stage 1 output |
| 3 | Propagation Analysis | Spark | Ready | Stage 2 output |
| 4 | Engagement Analysis | Spark | Ready | Stage 2 output |
| 5 | Temporal Analysis | Spark | Ready | Stage 2 output |
| 6 | NER | GPU | Ready | Stage 2 output |
| 7 | Sentiment Analysis | GPU | Ready | Stage 2 output |
| 8 | Topic Modeling | GPU | Ready | Stage 2 output |
| 9 | Event Classification | GPU | Ready | Stages 6,7,8 |
| 10 | Sustain/Decay Prediction | GPU | Ready | Stages 1,2,7 |
| 11 | Event Forecasting | GPU | Ready | Stage 1, ground truth |

**Legend:** Ready = Code complete, awaiting execution

### 5.3 What Needs to Be Finished

| Task | Priority | Platform | Estimated Time |
|------|----------|----------|----------------|
| Run Stage 1: Data Aggregation | High | RunPod GPU | 45-90 minutes |
| Run Stage 2: Anomaly Detection | High | EC2 Spark | 5-10 minutes |
| Run Stages 3-5 | High | EC2 Spark | 25-45 minutes |
| Run Stages 6-8 (NLP) | High | RunPod GPU | 2.5-6.5 hours |
| Run Stages 9-11 (ML) | Medium | RunPod GPU | 30-60 minutes |
| Build Quarto Website | Low | EC2 | ~1 minute |

### 5.4 Pre-Execution Checklist

Before running the pipeline, ensure:

1. **AWS S3 Data Available**
   - Raw Reddit data copied to s3://reddit-event-prediction-1776283460/reddit-data/parquet/
   - Expected: ~100+ parquet files

2. **AWS Credentials Configured**
   - export AWS_ACCESS_KEY_ID="your-key"
   - export AWS_SECRET_ACCESS_KEY="your-secret"
   - export AWS_SESSION_TOKEN="your-token" (if using temporary credentials)

3. **Compute Resources Ready**
   - RunPod/Lightning.ai A100 80GB instance provisioned
   - EC2 t3.large instance running with Java 17

4. **Environment Setup Complete**
   - Run setup scripts on respective machines
   - Verify GPU access: python -c "import torch; print(torch.cuda.is_available())"

---

## 6. Estimated Timeline and Costs

### Total Runtime

| Phase | Stages | Time |
|-------|--------|------|
| Setup | Environment setup | ~15 min |
| Phase 1 | Stage 1 (GPU aggregation) | 45-90 min |
| Phase 2 | Stage 2 (anomaly detection) | 5-10 min |
| Phase 3 | Stages 3-5 (parallel on EC2) | 25-45 min |
| Phase 4 | Stages 6-8 (parallel on RunPod) | 2.5-6.5 hours |
| Phase 5 | Stages 9-11 (sequential on RunPod) | 30-60 min |
| **Total** | | **~4-8 hours** |

Note: Phases 3 and 4 can run concurrently on their respective machines.

### Cost Estimates

- **Lightning.ai A100 80GB**: ~$2-3/hour
- **EC2 t3.large**: ~$0.10/hour
- **S3 Storage**: ~$0.023/GB/month (478 GB = ~$11/month)
- **Estimated total compute cost**: $20-50 depending on run duration

---

## 7. Output Files Summary

After complete execution, the following files will be available on S3:

| File | Stage | Description |
|------|-------|-------------|
| hourly_counts.parquet | 1 | Hourly activity per subreddit |
| daily_counts.parquet | 1 | Daily aggregated counts |
| subreddit_stats.parquet | 1 | Per-subreddit summary statistics |
| anomaly_windows.parquet | 2 | Detected anomaly windows |
| propagation_events.parquet | 3 | Cross-subreddit event clusters |
| spike_profiles.parquet | 4 | Spike shape classifications |
| temporal_patterns.parquet | 5 | Temporal activity patterns |
| entities.parquet | 6 | Extracted named entities |
| entity_cooccurrence.parquet | 6 | Entity pair co-occurrences |
| sentiment.parquet | 7 | Sentiment metrics per window |
| topics.parquet | 8 | Dominant topic per window |
| topic_details.parquet | 8 | Topic word representations |
| classifications.parquet | 9 | Event category predictions |
| feature_importance.parquet | 9 | Feature importance rankings |
| sustain_predictions.parquet | 10 | Sustain/decay predictions |
| forecast_results.parquet | 11 | Event forecasting results |

---

## 8. Key Technical Innovations

1. **Hybrid GPU-CPU Architecture**
   - GPU for data-intensive NLP and ML tasks
   - Spark for distributed aggregation and graph analysis

2. **GPU-Accelerated NLP Pipeline**
   - BERTopic with cuML UMAP/HDBSCAN for scalable topic modeling
   - Transformer-based NER and sentiment on GPU

3. **Checkpointing and Resume Capability**
   - Stage 1 saves monthly checkpoints
   - Can resume interrupted runs without recomputation

4. **Memory-Efficient Processing**
   - Month-by-month data loading
   - Column pruning and immediate aggregation
   - Garbage collection between stages

---

## 9. Next Steps

1. **Execute Stage 1** on RunPod GPU to aggregate 478GB raw data
2. **Sync outputs to EC2** and run Stages 2-5 (Spark stages)
3. **Upload anomaly_windows.parquet** to S3 for GPU NLP stages
4. **Execute Stages 6-8** (can run in parallel on separate instances)
5. **Execute Stages 9-11** sequentially for classification and forecasting
6. **Generate Quarto website** for visualization and reporting

---

*Report generated for DATS 6450 - Group 3*
