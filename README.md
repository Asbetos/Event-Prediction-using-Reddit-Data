# Reddit as an Early Warning System

This repository contains our DATS 6450 final project on detecting, characterizing, and forecasting real-world events from Reddit activity at scale.

We process Reddit comments and submissions from June 2023 to July 2024 and use an 11-stage pipeline spanning:

- large-scale aggregation
- anomaly detection
- temporal and propagation analysis
- named entity extraction
- sentiment and topic modeling
- event classification and forecasting

## Repository Structure

```text
.
├── Code/
│   ├── analysis/          Offline analysis and asset-generation scripts
│   ├── config/            Shared settings and Spark configuration
│   ├── data/
│   │   ├── ground_truth/  Curated event labels
│   │   └── intermediate/  Stage output parquet artifacts
│   ├── outputs/           Generated numeric summaries
│   ├── pipeline/          Spark / CPU stages
│   ├── runpod/            GPU stages and orchestration scripts
│   └── utils/             Shared helpers
│
├── Final-Project-Report/
│   ├── Final-Project-Report.pdf
│   └── figures/report/    Figures used in the final report
│
├── Final-Project-Presentation/
│   ├── presentation.qmd
│   ├── presentation.css
│   ├── presentation.html
│   ├── presentation_assets/
│   └── presentation_files/
│
├── Final-Project-Proposal/
│   └── project_proposal_group3.pdf
│
└── setup.sh
```

## Final Deliverables

- Final report: `Final-Project-Report/Final-Project-Report.pdf`
- Final presentation source: `Final-Project-Presentation/presentation.qmd`
- Final presentation render: `Final-Project-Presentation/presentation.html`

## Quick Start

### 1. Offline analysis and figure regeneration

Run these from the repository root:

```bash
python Code/analysis/summarize_results.py
python Code/analysis/generate_eda_figures.py
python Code/analysis/generate_report_figures.py
python Code/analysis/generate_presentation_assets.py
```

These scripts read from:

- `Code/data/intermediate/`
- `Code/data/ground_truth/events.csv`

and write to:

- `Code/outputs/stage_results.json`
- `Final-Project-Report/figures/report/`
- `Final-Project-Presentation/presentation_assets/`

### 2. Render the presentation

```bash
quarto render Final-Project-Presentation/presentation.qmd
```

or:

```bash
cd Final-Project-Presentation
quarto render presentation.qmd
```

## Headline Results

| Metric | Value |
|---|---|
| Comments processed | **1.19 B** |
| Submissions processed | **87.2 M** |
| Hourly subreddit rows | **9.38 M** |
| Anomaly windows detected | **31,938** across **499 / 500** subreddits |
| Stage 6 entities extracted | **63,614** rows / **36,082** unique entities / **1,738** windows |
| Stage 11 forecasting AUC | **0.955** (5-fold CV); F1 **0.77**; precision **0.85**; recall **0.70** |
| Stage 10 sustain AUC | **0.949**; positive precision/recall = 0 under extreme imbalance |

## Ground Truth

The curated event set contains 35 events across five categories:

| Category | Count |
|---|---:|
| Breaking news | 13 |
| Controversy | 7 |
| Product launch | 7 |
| Disaster | 5 |
| Meme / viral | 3 |

Ground-truth file: `Code/data/ground_truth/events.csv`

## Full Pipeline Execution

The full pipeline still assumes the original split between local CPU / Spark stages and GPU stages.

### Environment setup

```bash
chmod +x setup.sh && ./setup.sh
chmod +x Code/runpod/setup_runpod.sh && ./Code/runpod/setup_runpod.sh
```

### Stage execution examples

From the repository root:

```bash
python Code/runpod/stage1_aggregate_gpu.py
python -m compileall Code
```

From inside `Code/`:

```bash
python -m pipeline.stage2_anomaly_detection
python -m pipeline.stage3_propagation
python -m pipeline.stage4_engagement
python -m pipeline.stage5_temporal
bash runpod/run_all_gpu.sh
python -m pipeline.stage_ml_spark
```

## Notes

- The `Code/analysis` scripts now write report outputs to `Final-Project-Report/` and presentation outputs to `Final-Project-Presentation/`.
- The raw cloud pipeline stages in `Code/pipeline/` and `Code/runpod/` still depend on the appropriate local environment, Spark, GPU libraries, and/or cloud credentials.

## License

MIT License

