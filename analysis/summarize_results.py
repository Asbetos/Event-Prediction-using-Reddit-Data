#!/usr/bin/env python3
"""Compute and dump the full numeric summary used in RESULTS.md and FINAL_REPORT.md.

Reads everything from data/intermediate/ and writes stage_results.json next to it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "intermediate"
OUT = ROOT / "stage_results.json"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def topn(s: pd.Series, n: int = 10) -> dict:
    return s.value_counts().head(n).to_dict()


def main() -> None:
    results: dict = {}

    # --- Stage 1 ---
    hourly = pd.read_parquet(D / "hourly_counts.parquet")
    daily = pd.read_parquet(D / "daily_counts.parquet")
    subs = pd.read_parquet(D / "subreddit_stats.parquet")
    results["stage1"] = {
        "hourly_rows": int(len(hourly)),
        "daily_rows": int(len(daily)),
        "subreddit_stat_rows": int(len(subs)),
        "unique_subreddits": int(hourly["subreddit"].nunique()),
        "date_min": str(hourly["hour_bucket"].min()),
        "date_max": str(hourly["hour_bucket"].max()),
        "comments_total": int(hourly[hourly["data_type"] == "comments"]["post_count"].sum()),
        "submissions_total": int(hourly[hourly["data_type"] == "submissions"]["post_count"].sum()),
    }

    # --- Stage 2 ---
    aw = pd.read_parquet(D / "anomaly_windows.parquet")
    results["stage2"] = {
        "anomaly_windows": int(len(aw)),
        "unique_subreddits_with_anomaly": int(aw["subreddit"].nunique()),
        "avg_duration_hours": float(aw["duration_hours"].mean()),
        "median_duration_hours": float(aw["duration_hours"].median()),
        "max_peak_z": float(aw["peak_z_score"].max()),
        "median_peak_z": float(aw["peak_z_score"].median()),
        "p95_peak_z": float(aw["peak_z_score"].quantile(0.95)),
        "avg_peak_z": float(aw["peak_z_score"].mean()),
        "avg_anomaly_hours": float(aw["anomaly_hours"].mean()),
        "top10_subreddits": topn(aw["subreddit"], 10),
    }

    # --- Stage 3 ---
    prop = pd.read_parquet(D / "propagation_events.parquet")
    results["stage3"] = {
        "n_clusters": int(len(prop)),
        "propagation_types": prop["propagation_type"].value_counts().to_dict(),
        "max_subreddits_per_cluster": int(prop["num_subreddits"].max()),
        "avg_duration_hours": float(prop["total_duration_hours"].mean()),
    }

    # --- Stage 4 ---
    sp = pd.read_parquet(D / "spike_profiles.parquet")
    valid = sp.dropna(subset=["peak_z_score", "post_spike_avg_score"])
    valid = valid[valid["post_spike_avg_score"] > 0]
    if len(valid) > 5:
        pearson_r, pearson_p = scipy_stats.pearsonr(valid["peak_z_score"], valid["post_spike_avg_score"])
        spearman_r, spearman_p = scipy_stats.spearmanr(valid["peak_z_score"], valid["post_spike_avg_score"])
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")
    er = sp["engagement_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    results["stage4"] = {
        "spike_profile_rows": int(len(sp)),
        "shape_distribution": sp["spike_shape"].value_counts().to_dict(),
        "pearson_r_z_vs_engagement": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r_z_vs_engagement": float(spearman_r),
        "spearman_p": float(spearman_p),
        "avg_engagement_ratio": float(er.mean()),
        "median_engagement_ratio": float(er.median()),
    }

    # --- Stage 5 ---
    tp = pd.read_parquet(D / "temporal_patterns.parquet")
    peak = tp.loc[tp["anomaly_count"].idxmax()]
    results["stage5"] = {
        "rows": int(len(tp)),
        "peak_dow": DOW[int(peak["day_of_week"])],
        "peak_hour": int(peak["hour_of_day"]),
        "peak_anomaly_count": int(peak["anomaly_count"]),
        "weekday_anomalies": int(tp[tp["day_of_week"].between(0, 4)]["anomaly_count"].sum()),
        "weekend_anomalies": int(tp[tp["day_of_week"].between(5, 6)]["anomaly_count"].sum()),
    }

    # --- Stage 6 ---
    ents = pd.read_parquet(D / "entities.parquet")
    results["stage6"] = {
        "entity_rows": int(len(ents)),
        "windows_with_entities": int(ents["window_id"].nunique()),
        "subreddits_with_entities": int(ents["subreddit"].nunique()),
        "label_distribution": ents["entity_label"].value_counts().to_dict(),
        "unique_entities": int(ents["entity_text"].nunique()),
        "top10_PERSON": ents[ents["entity_label"] == "PERSON"].groupby("entity_text")["count"].sum().nlargest(10).to_dict(),
        "top10_ORG": ents[ents["entity_label"] == "ORG"].groupby("entity_text")["count"].sum().nlargest(10).to_dict(),
        "top10_GPE": ents[ents["entity_label"] == "GPE"].groupby("entity_text")["count"].sum().nlargest(10).to_dict(),
    }

    # --- Stage 7 ---
    sent = pd.read_parquet(D / "sentiment.parquet")
    results["stage7"] = {
        "rows": int(len(sent)),
        "windows": int(sent["window_id"].nunique()),
        "subreddits": int(sent["subreddit"].nunique()),
        "n_significant_shifts": int(sent["shift_significant"].astype(bool).sum()),
        "mean_sentiment_shift": float(sent["sentiment_shift"].mean()),
    }

    # --- Stage 8 ---
    topics = pd.read_parquet(D / "topics.parquet")
    topic_det = pd.read_parquet(D / "topic_details.parquet")
    results["stage8"] = {
        "windows_with_topic": int(len(topics)),
        "topic_id_dist": topics["dominant_topic"].value_counts().to_dict(),
        "n_topics_overall": int(len(topic_det)),
        "topic_details": topic_det.to_dict(orient="records"),
    }

    # --- Stage 9 ---
    clf = pd.read_parquet(D / "classifications.parquet")
    results["stage9"] = {
        "n_predictions": int(len(clf)),
        "category_distribution": clf["predicted_category"].value_counts().to_dict(),
    }

    # --- Stage 10 ---
    sus = pd.read_parquet(D / "sustain_predictions.parquet")
    sus_imp = pd.read_parquet(D / "sustain_feature_importance.parquet")
    results["stage10"] = {
        "n_predictions": int(len(sus)),
        "label_distribution": sus["sustain_label"].value_counts().to_dict(),
        "predicted_label_distribution": sus["predicted_label"].value_counts().to_dict(),
        "model_roc_auc": float(sus["model_roc_auc"].iloc[0]),
        "model_precision": float(sus["model_precision"].iloc[0]),
        "model_recall": float(sus["model_recall"].iloc[0]),
        "model_f1": float(sus["model_f1"].iloc[0]),
        "feature_importance": sus_imp.sort_values("importance", ascending=False).to_dict(orient="records"),
    }

    # --- Stage 11 ---
    fc = pd.read_parquet(D / "forecast_results.parquet")
    fc_imp = pd.read_parquet(D / "forecast_feature_importance.parquet")
    fc_fold = pd.read_parquet(D / "forecast_fold_metrics.parquet")
    results["stage11"] = {
        "n_predictions": int(len(fc)),
        "n_positive": int(fc["label"].sum()),
        "n_negative": int((fc["label"] == 0).sum()),
        "model_roc_auc": float(fc["model_roc_auc"].iloc[0]),
        "model_avg_precision": float(fc["model_avg_precision"].iloc[0]),
        "model_precision": float(fc["model_precision"].iloc[0]),
        "model_recall": float(fc["model_recall"].iloc[0]),
        "model_f1": float(fc["model_f1"].iloc[0]),
        "fold_metrics": fc_fold.to_dict(orient="records"),
        "feature_importance": fc_imp.sort_values("importance", ascending=False).to_dict(orient="records"),
    }

    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
