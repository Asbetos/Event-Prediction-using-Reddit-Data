# Handoff

## Repo State

- Repo: `/home/ubuntu/Event-Prediction-using-Reddit-Data`
- Branch: `main`
- Latest pushed commit: `c0340e4` (`Harden pipeline execution and split S3 storage`)
- Generated artifacts were not pushed to GitHub.

## Code Changes Already Made

- Split raw and processed S3 bucket config in `config/settings.py`.
- Hardened RunPod setup and Stage 1 execution.
- Increased local Spark timeout settings in `config/spark_config.py`.
- Fixed syntax bug in `pipeline/stage3_propagation.py`.

## Pipeline Status

### Completed

- Stage 1: complete on RunPod
- Stage 2: complete locally
- Stage 3: complete locally
- Stage 4: complete locally
- Stage 5: complete locally

### Local Outputs Present

- `data/intermediate/hourly_counts.parquet`
- `data/intermediate/daily_counts.parquet`
- `data/intermediate/subreddit_stats.parquet`
- `data/intermediate/anomaly_windows.parquet/`
- `data/intermediate/propagation_events.parquet/`
- `data/intermediate/spike_profiles.parquet/`
- `data/intermediate/temporal_patterns.parquet/`

### Processed S3 Outputs Present

Under `s3://reddit-event-prediction-147390571732-processed-20260423/reddit-data/intermediate/`:

- `hourly_counts.parquet`
- `daily_counts.parquet`
- `subreddit_stats.parquet`
- `stage1_checkpoint/`
- `anomaly_windows.parquet/`
- `propagation_events.parquet/`
- `spike_profiles.parquet/`
- `temporal_patterns.parquet/`

## Stage Summaries So Far

### Stage 2

- Anomaly windows: `31,938`
- Unique subreddits: `499`
- Avg duration: `1.7` hours
- Max peak z-score: `3540.33`

### Stage 3

- Event clusters: `1`
- Co-occurring pairs: `5,151,593`
- Propagation type result: `niche_to_mainstream`

### Stage 4

- Spike profiles: `31,938`
- Distribution:
  - `double_peak: 31,673`
  - `slow_burn: 196`
  - `sharp_spike: 64`
  - `sustained_plateau: 5`

### Stage 5

- Temporal pattern rows: `168`
- Peak anomaly slot: `Tue 16:00 UTC`
- Ground truth events loaded: `35`

## Remaining Project Work

### RunPod GPU Stages

1. Stage 6: `runpod/stage6_ner_gpu.py`
2. Stage 7: `runpod/stage7_sentiment_gpu.py`
3. Stage 8: `runpod/stage8_topics_gpu.py`

These need:
- access to raw S3 data bucket
- access to processed S3 bucket
- `anomaly_windows.parquet` already in processed S3

Expected outputs:
- `entities.parquet`
- `entity_cooccurrence.parquet`
- `sentiment.parquet`
- `topics.parquet`
- `topic_details.parquet`

### Final Modeling Stages

4. Stage 9: `runpod/stage9_classification_gpu.py`
5. Stage 10: `runpod/stage10_sustain_gpu.py`
6. Stage 11: `runpod/stage11_forecast_gpu.py`

Expected outputs:
- `classifications.parquet`
- `feature_importance.parquet`
- `sustain_predictions.parquet`
- `sustain_feature_importance.parquet`
- `forecast_results.parquet`
- `forecast_feature_importance.parquet`
- `forecast_fold_metrics.parquet`

### Optional Course Deliverable

7. Spark baseline: `python -m pipeline.stage_ml_spark`

### Final Presentation Artifact

8. Build website / final render as needed

## Recommended Execution Order

1. Run Stage 6
2. Run Stage 7
3. Run Stage 8
4. Run Stage 9
5. Run Stage 10
6. Run Stage 11
7. Optionally run Spark baseline
8. Render/update website

If multiple GPU pods are available:
- run Stages 6, 7, 8 in parallel

If only one GPU pod is available:
- run Stages 6, 7, 8 sequentially

## Useful Commands For Next Agent

### Check current repo state

```bash
git status --short --branch
```

### Check local stage logs

```bash
python3 - <<'PY'
from pathlib import Path
for name in ['logs/stage2.log', 'logs/stage3.log', 'logs/stage4.log', 'logs/stage5.log']:
    p = Path(name)
    if p.exists():
        print(f'===== {name} =====')
        print(p.read_text(errors='ignore')[-3000:])
PY
```

### Check processed S3 contents

```bash
aws s3 ls s3://reddit-event-prediction-147390571732-processed-20260423/reddit-data/intermediate/
```

### Run the next GPU stage

```bash
python runpod/stage6_ner_gpu.py
```

## Main Risk / Context

- AWS Academy lab sessions are time-limited and credentials are temporary.
- The raw and processed buckets are split:
  - raw source bucket: `reddit-event-prediction-1776283460`
  - processed bucket: `reddit-event-prediction-147390571732-processed-20260423`
- Any future agent should verify current AWS credentials before launching GPU stages.
