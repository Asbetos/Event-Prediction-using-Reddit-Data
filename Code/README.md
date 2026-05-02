# Code README

This folder contains the runnable project code, staged data artifacts, and analysis scripts for the Reddit event prediction pipeline.

## Structure

```text
Code/
├── analysis/   Offline summaries, report figures, and presentation asset generation
├── config/     Shared constants, local/S3 paths, Spark configuration
├── data/
│   ├── ground_truth/   Curated event labels
│   └── intermediate/   Stage output parquet artifacts used by offline analysis
├── outputs/    Generated numeric summaries
├── pipeline/   Spark / CPU stages
├── runpod/     GPU stages and orchestration scripts
└── utils/      Shared helpers
```

## What is in here

- `analysis/` is the easiest place to start if you just want to regenerate results from the checked-in parquet outputs.
- `pipeline/` contains the Spark-based stages for anomaly detection, propagation, engagement, and temporal analysis.
- `runpod/` contains the GPU-heavy stages for aggregation, NER, sentiment, topics, classification, sustain prediction, and forecasting.

## Running Offline Analysis

These commands assume you are in the `Code/` directory.

```bash
python analysis/summarize_results.py
python analysis/generate_eda_figures.py
python analysis/generate_report_figures.py
python analysis/generate_presentation_assets.py
```

Outputs are written to:

- `outputs/stage_results.json`
- `../Final-Project-Report/figures/report/`
- `../Final-Project-Presentation/presentation_assets/`

## Running Spark Stages

Run these from inside `Code/`:

```bash
python -m pipeline.stage2_anomaly_detection
python -m pipeline.stage3_propagation
python -m pipeline.stage4_engagement
python -m pipeline.stage5_temporal
python -m pipeline.stage_ml_spark
```

## Running GPU Stages

Run these from inside `Code/`:

```bash
python runpod/stage1_aggregate_gpu.py
bash runpod/run_all_gpu.sh
```

Or run individual GPU stages directly as needed.

## Environment Notes

- `pipeline/` requires `pyspark` to be installed.
- `runpod/` stages require the intended GPU environment, including RAPIDS / cuML / cuDF and the NLP model dependencies.
- Some cloud execution paths also require the appropriate S3 credentials and environment variables from `config/settings.py`.

## Data Notes

- `data/intermediate/` is the local stage-output contract used by the offline scripts.
- The analysis scripts are designed so you can regenerate figures and summaries without rerunning the full cloud pipeline.

## Final Deliverables Using This Code

- Report figures: `../Final-Project-Report/figures/report/`
- Presentation assets: `../Final-Project-Presentation/presentation_assets/`

If you are starting from scratch, begin with `analysis/summarize_results.py` and `analysis/generate_report_figures.py` to confirm the local data and dependencies are working.
