# Execution Plan: Reddit Event Prediction Pipeline on Lightning.ai

## Overview
This plan details running the 11-stage pipeline on Lightning.ai A100 80GB GPU instance.

---

## Prerequisites

### 1. S3 Data Transfer (Wait for completion)
```bash
# Check copy progress
tail -f /tmp/s3_copy.log

# Verify completion
aws s3 ls s3://reddit-event-prediction-1776283460/reddit-data/parquet/ --recursive | wc -l
```

### 2. Lightning.ai Setup
- **Instance**: A100 80GB (or A100 40GB if 80GB unavailable)
- **Region**: us-east-1 (same as S3 bucket for lower latency)
- **OS**: Ubuntu 22.04 or 24.04
- **Storage**: 100GB+ for caching intermediate data locally

---

## Phase 1: Lightning.ai Environment Setup (15-30 min)

### Step 1.1: Create Lightning.ai Instance
1. Go to https://lightning.ai
2. Create new studio with A100 GPU
3. Select PyTorch/CUDA base image

### Step 1.2: Clone Repository
```bash
cd /home/ubuntu
git clone https://github.com/Asbetos/Event-Prediction-using-Reddit-Data.git
cd Event-Prediction-using-Reddit-Data
```

### Step 1.3: Set AWS Credentials
```bash
export AWS_ACCESS_KEY_ID="<your-access-key-id>"
export AWS_SECRET_ACCESS_KEY="<your-secret-access-key>"
export AWS_SESSION_TOKEN="<your-session-token>"  # if using temporary credentials
```

### Step 1.4: Run Setup Script
```bash
chmod +x setup_lightning.sh
./setup_lightning.sh
```

---

## Phase 2: GPU Stages (Run on Lightning.ai)

### Stage 1: Data Aggregation (45-90 min)
**Reads 478 GB raw Reddit data and produces aggregated counts.**

```bash
cd /home/ubuntu/Event-Prediction-using-Reddit-Data
python runpod/stage1_aggregate_gpu.py
```

**Outputs to S3:**
- `hourly_counts.parquet`
- `daily_counts.parquet`
- `subreddit_stats.parquet`

**Checkpointing:** If interrupted, re-run to resume from last completed month.

---

## Phase 3: EC2 Spark Stages (Requires EC2 instance)

> **Note:** Stages 2-5 require PySpark on EC2 (t3.large). These cannot run on Lightning.ai without Spark.

### Option A: Use EC2 Instance
1. Launch EC2 t3.large in us-east-1
2. Install Java 17, PySpark
3. Run `setup.sh`
4. Sync intermediate data from S3
5. Run stages 2-5

### Option B: Run Spark Locally (Slower)
If Lightning.ai has sufficient resources, can attempt local Spark:
```bash
pip install pyspark==3.5.5
# May need to adjust memory settings in spark_config.py
```

**Stage 2: Anomaly Detection (5-10 min)**
```bash
# Download Stage 1 outputs first
aws s3 sync s3://reddit-event-prediction-1776283460/reddit-data/intermediate/ \
    data/intermediate/ \
    --exclude "*" \
    --include "hourly_counts.parquet/*" \
    --include "daily_counts.parquet/*" \
    --include "subreddit_stats.parquet/*"

python -m pipeline.stage2_anomaly_detection
```

**Stage 3: Propagation Analysis (5-10 min)**
```bash
python -m pipeline.stage3_propagation
```

**Stage 4: Engagement Analysis (10-15 min)**
```bash
python -m pipeline.stage4_engagement
```

**Stage 5: Temporal Analysis (5-10 min)**
```bash
python -m pipeline.stage5_temporal
```

**After Stage 2-5, upload to S3:**
```bash
aws s3 cp data/intermediate/anomaly_windows.parquet \
    s3://reddit-event-prediction-1776283460/reddit-data/intermediate/ --recursive
aws s3 cp data/intermediate/propagation_events.parquet \
    s3://reddit-event-prediction-1776283460/reddit-data/intermediate/ --recursive
aws s3 cp data/intermediate/spike_profiles.parquet \
    s3://reddit-event-prediction-1776283460/reddit-data/intermediate/ --recursive
aws s3 cp data/intermediate/temporal_patterns.parquet \
    s3://reddit-event-prediction-1776283460/reddit-data/intermediate/ --recursive
```

---

## Phase 4: GPU NLP Stages (Run on Lightning.ai, 2-4 hours each)

> **Requires:** `anomaly_windows.parquet` in S3 from Stage 2

### Stage 6: Named Entity Recognition (2-4 hours)
```bash
python runpod/stage6_ner_gpu.py
```

### Stage 7: Sentiment Analysis (2-4 hours)
```bash
python runpod/stage7_sentiment_gpu.py
```

### Stage 8: Topic Modeling (30-90 min)
```bash
python runpod/stage8_topics_gpu.py
```

> **These stages can run in parallel** on separate Lightning.ai instances if you have multiple.

---

## Phase 5: Classification & Prediction (Run on Lightning.ai, 30-60 min)

> **Requires:** Outputs from Stages 6, 7, 8 in S3

### Stage 9: Event Classification (10-20 min)
```bash
python runpod/stage9_classification_gpu.py
```

### Stage 10: Sustain/Decay Prediction (10-20 min)
```bash
python runpod/stage10_sustain_gpu.py
```

### Stage 11: Event Forecasting (10-20 min)
```bash
python runpod/stage11_forecast_gpu.py
```

---

## Total Estimated Timeline

| Phase | Stages | Platform | Time |
|-------|--------|----------|------|
| Data Copy | - | S3 | 4-6 hours |
| Setup | - | Lightning.ai | 15-30 min |
| Phase 1 | Stage 1 | Lightning.ai GPU | 45-90 min |
| Phase 2-3 | Stages 2-5 | EC2 Spark | 25-45 min |
| Phase 4 | Stages 6-8 | Lightning.ai GPU | 2.5-6.5 hours |
| Phase 5 | Stages 9-11 | Lightning.ai GPU | 30-60 min |

**Total: ~8-14 hours** (including data copy)

---

## Monitoring & Debugging

### Check Logs
```bash
tail -f /workspace/logs/stage1_aggregate.log
tail -f /workspace/logs/stage6_ner.log
# etc.
```

### Check S3 Outputs
```bash
aws s3 ls s3://reddit-event-prediction-1776283460/reddit-data/intermediate/
```

### GPU Memory Check
```python
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f} / {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB")
```

---

## Cost Estimates

- **Lightning.ai A100 80GB**: ~$2-3/hour
- **EC2 t3.large**: ~$0.10/hour
- **S3 Storage**: ~$0.023/GB/month (478 GB = ~$11/month)
- **S3 Data Transfer**: Free within same region

**Estimated total compute cost**: $20-50 depending on run duration

---

## Notes

1. **Session Token Expiry**: AWS session tokens expire. If stages take long, may need to refresh credentials.

2. **Checkpointing**: Stage 1 supports checkpointing. If it fails, re-run to resume.

3. **Parallel Execution**: Stages 6-8 can run in parallel on 3 separate Lightning.ai instances to save time.

4. **Spark Alternative**: If no EC2 available, consider using Amazon EMR Serverless or AWS Glue for Stages 2-5.
