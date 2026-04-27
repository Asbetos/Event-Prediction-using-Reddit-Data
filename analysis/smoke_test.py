"""Quick sanity check that all stage parquets and the JSON summary are present and consistent."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

results = json.loads((ROOT / "stage_results.json").read_text())
print(f"stage_results.json keys: {list(results.keys())}")

# Spot-check a few headline numbers
print(f"  Stage 1  hourly rows  : {results['stage1']['hourly_rows']:,}")
print(f"  Stage 2  anomaly wins : {results['stage2']['anomaly_windows']:,}")
print(f"  Stage 6  entity rows  : {results['stage6']['entity_rows']:,}")
print(f"  Stage 11 AUC          : {results['stage11']['model_roc_auc']:.4f}")
print(f"  Stage 11 F1           : {results['stage11']['model_f1']:.4f}")

# Verify every parquet is readable
expected = [
    "hourly_counts", "daily_counts", "subreddit_stats",
    "anomaly_windows", "propagation_events", "spike_profiles",
    "temporal_patterns", "entities", "sentiment", "topics",
    "topic_details", "classifications", "sustain_predictions",
    "sustain_feature_importance", "forecast_results",
    "forecast_feature_importance", "forecast_fold_metrics",
]
for name in expected:
    p = ROOT / "data" / "intermediate" / f"{name}.parquet"
    if not p.exists():
        print(f"  MISSING: {p}")
        continue
    df = pd.read_parquet(p)
    print(f"  OK  {name:32s} {df.shape}")

# Verify every figure is present
expected_figs = [
    "anomaly_heatmap_dow_hour.png", "anomaly_polar_hour.png",
    "classification_predicted_categories.png", "entities_label_distribution.png",
    "entities_top_by_type.png", "event_category_by_dow.png",
    "forecast_events_per_label.png", "forecast_summary.png",
    "monthly_anomaly_counts.png", "pipeline_output_volume.png",
    "propagation_network_top10.png", "propagation_scatter.png",
    "propagation_type_distribution.png", "sentiment_class_proportions.png",
    "spike_magnitude_vs_engagement.png", "spike_shape_distribution.png",
    "spike_shape_examples.png", "sustain_confusion_and_importance.png",
    "topics_distribution.png", "zscore_distribution.png",
]
for f in expected_figs:
    p = ROOT / "website" / "figures" / f
    print(f"  {'OK ' if p.exists() else 'MISS'} {f}")

print("Smoke test done.")
