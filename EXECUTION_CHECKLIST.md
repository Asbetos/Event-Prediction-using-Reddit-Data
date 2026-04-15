# Pipeline Execution Checklist

## Pre-flight Checks

### 1. Verify S3 Data Copy Complete
```bash
# Check if copy process is still running
ps aux | grep "aws s3 cp" | grep -v grep

# If still running, monitor progress
tail -f /tmp/s3_copy.log

# Verify file count when done
aws s3 ls s3://reddit-event-prediction-1776283460/reddit-data/parquet/ --recursive | wc -l
# Expected: ~100+ parquet files
```

### 2. Verify AWS Credentials
```bash
aws sts get-caller-identity
# Should return your identity without errors
```

---

## Lightning.ai Setup

### Step 1: Create Instance
- [ ] Login to https://lightning.ai
- [ ] Create new Studio
- [ ] Select A100 80GB GPU
- [ ] Region: us-east-1
- [ ] Storage: 100GB+

### Step 2: Environment Setup
- [ ] Clone repository
```bash
cd /home/ubuntu
git clone https://github.com/Asbetos/Event-Prediction-using-Reddit-Data.git
cd Event-Prediction-using-Reddit-Data
```

- [ ] Set AWS credentials
```bash
export AWS_ACCESS_KEY_ID="ASIASEUJJUTKGSUPS7X7"
export AWS_SECRET_ACCESS_KEY="l9KdUMVEfKQST09gLlv40Y0pUwVDi4ADL4p/OKnm"
export AWS_SESSION_TOKEN="IQoJb3JpZ2luX2VjEOP..."
```

- [ ] Run setup script
```bash
chmod +x setup_lightning.sh
./setup_lightning.sh
```

- [ ] Verify GPU access
```python
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Stage Execution

### Stage 1: Data Aggregation (45-90 min)
- [ ] Run aggregation
```bash
cd /home/ubuntu/Event-Prediction-using-Reddit-Data
python runpod/stage1_aggregate_gpu.py
```
- [ ] Verify outputs in S3
```bash
aws s3 ls s3://reddit-event-prediction-1776283460/reddit-data/intermediate/
# Should show: hourly_counts.parquet/, daily_counts.parquet/, subreddit_stats.parquet/
```

---

### Stages 2-5: EC2 Spark (Requires EC2)

#### Option A: Launch EC2 t3.large
- [ ] Launch EC2 t3.large in us-east-1
- [ ] Install dependencies
```bash
sudo apt-get update
sudo apt-get install -y openjdk-17-jdk python3-pip
pip3 install --user pyspark pandas pyarrow boto3 s3fs
```

- [ ] Clone repo and sync from S3
```bash
git clone https://github.com/Asbetos/Event-Prediction-using-Reddit-Data.git
cd Event-Prediction-using-Reddit-Data
aws s3 sync s3://reddit-event-prediction-1776283460/reddit-data/intermediate/ \
    data/intermediate/ \
    --exclude "*" \
    --include "hourly_counts.parquet/*" \
    --include "daily_counts.parquet/*" \
    --include "subreddit_stats.parquet/*"
```

#### Stage 2: Anomaly Detection (5-10 min)
- [ ] Run anomaly detection
```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
python3 -m pipeline.stage2_anomaly_detection
```
- [ ] Verify output
```bash
ls -la data/intermediate/anomaly_windows.parquet/
```

#### Stage 3: Propagation (5-10 min)
- [ ] Run propagation analysis
```bash
python3 -m pipeline.stage3_propagation
```
- [ ] Verify output
```bash
ls -la data/intermediate/propagation_events.parquet/
```

#### Stage 4: Engagement (10-15 min)
- [ ] Run engagement analysis
```bash
python3 -m pipeline.stage4_engagement
```
- [ ] Verify output
```bash
ls -la data/intermediate/spike_profiles.parquet/
```

#### Stage 5: Temporal (5-10 min)
- [ ] Run temporal analysis
```bash
python3 -m pipeline.stage5_temporal
```
- [ ] Verify output
```bash
ls -la data/intermediate/temporal_patterns.parquet/
```

#### Upload EC2 outputs to S3
- [ ] Sync outputs
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

### Stages 6-8: NLP (Run on Lightning.ai)

#### Stage 6: NER (2-4 hours)
- [ ] Run NER extraction
```bash
python runpod/stage6_ner_gpu.py
```
- [ ] Monitor progress
```bash
tail -f /workspace/logs/stage6_ner.log
```
- [ ] Verify output in S3
```bash
aws s3 ls s3://reddit-event-prediction-1776283460/reddit-data/intermediate/entities.parquet/
```

#### Stage 7: Sentiment (2-4 hours)
- [ ] Run sentiment analysis
```bash
python runpod/stage7_sentiment_gpu.py
```
- [ ] Monitor progress
```bash
tail -f /workspace/logs/stage7_sentiment.log
```
- [ ] Verify output in S3
```bash
aws s3 ls s3://reddit-event-prediction-1776283460/reddit-data/intermediate/sentiment.parquet/
```

#### Stage 8: Topics (30-90 min)
- [ ] Run topic modeling
```bash
python runpod/stage8_topics_gpu.py
```
- [ ] Monitor progress
```bash
tail -f /workspace/logs/stage8_topics.log
```
- [ ] Verify output in S3
```bash
aws s3 ls s3://reddit-event-prediction-1776283460/reddit-data/intermediate/topics.parquet/
```

---

### Stages 9-11: Classification & Prediction

#### Stage 9: Classification (10-20 min)
- [ ] Run event classification
```bash
python runpod/stage9_classification_gpu.py
```
- [ ] Verify output
```bash
aws s3 ls s3://reddit-event-prediction-1776283460/reddit-data/intermediate/classifications.parquet/
```

#### Stage 10: Sustain Prediction (10-20 min)
- [ ] Run sustain prediction
```bash
python runpod/stage10_sustain_gpu.py
```
- [ ] Verify output
```bash
aws s3 ls s3://reddit-event-prediction-1776283460/reddit-data/intermediate/sustain_predictions.parquet/
```

#### Stage 11: Forecasting (10-20 min)
- [ ] Run event forecasting
```bash
python runpod/stage11_forecast_gpu.py
```
- [ ] Verify output
```bash
aws s3 ls s3://reddit-event-prediction-1776283460/reddit-data/intermediate/forecast_results.parquet/
```

---

## Final Verification

### Check All Outputs
```bash
aws s3 ls s3://reddit-event-prediction-1776283460/reddit-data/intermediate/
```

Expected files:
- [ ] `hourly_counts.parquet/`
- [ ] `daily_counts.parquet/`
- [ ] `subreddit_stats.parquet/`
- [ ] `anomaly_windows.parquet/`
- [ ] `propagation_events.parquet/`
- [ ] `spike_profiles.parquet/`
- [ ] `temporal_patterns.parquet/`
- [ ] `entities.parquet/`
- [ ] `entity_cooccurrence.parquet/`
- [ ] `sentiment.parquet/`
- [ ] `topics.parquet/`
- [ ] `topic_details.parquet/`
- [ ] `classifications.parquet/`
- [ ] `sustain_predictions.parquet/`
- [ ] `forecast_results.parquet/`

---

## Troubleshooting

### CUDA Out of Memory
```python
import torch
torch.cuda.empty_cache()
# Or reduce batch sizes in the scripts
```

### AWS Credentials Expired
```bash
# Re-export credentials
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
```

### Stage 1 Checkpoint Resume
```bash
# If Stage 1 fails, check which months are complete
aws s3 ls s3://reddit-event-prediction-1776283460/reddit-data/intermediate/stage1_checkpoint/

# Re-run to resume
python runpod/stage1_aggregate_gpu.py
```
