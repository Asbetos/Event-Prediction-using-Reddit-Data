#!/usr/bin/env python3
"""
Stage 10: Spike Sustain vs Decay Prediction (Q9)
===================================================
Predicts whether an activity spike will be sustained (>2x baseline for >24h)
or decay quickly, using features from the first 2-4 hours of each spike.

Uses cuML RandomForest with time-based train/test split.

Outputs:
  - sustain_predictions.parquet: predictions with ROC-AUC, precision, recall

Usage: python stage10_sustain_gpu.py
"""

import os
import sys
import time
import logging
import gc
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import s3fs
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

# ── GPU imports with fallback ───────────────────────────────────────────────
try:
    import cudf
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    GPU_ML = True
    print("cuML available - using GPU RandomForest")
except ImportError:
    GPU_ML = False
    from sklearn.ensemble import RandomForestClassifier
    print("cuML not available - using sklearn RandomForest")

# Allow CPU-only mode when GPU is shared with NLP stages (run_all_gpu.sh)
if os.environ.get("FORCE_CPU", "0") == "1":
    GPU_ML = False
    from sklearn.ensemble import RandomForestClassifier
    print("FORCE_CPU=1: Using sklearn on EPYC cores to avoid GPU contention")

# ── Logging ─────────────────────────────────────────────────────────────────
os.makedirs("/workspace/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/workspace/logs/stage10_sustain.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
S3_BUCKET = "ven-bda-s3-v2"
S3_INTERMEDIATE = f"s3://{S3_BUCKET}/reddit-data/intermediate"

SUSTAIN_THRESHOLD_MULTIPLIER = 2.0  # >2x baseline
SUSTAIN_DURATION_HOURS = 24         # for >24 hours
EARLY_WINDOW_HOURS = 4              # features from first N hours


def get_s3_storage_options():
    """Build s3fs storage options from environment or ~/.aws/credentials."""
    opts = {}
    key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    token = os.environ.get("AWS_SESSION_TOKEN", "")
    if key and secret:
        opts["key"] = key
        opts["secret"] = secret
        if token:
            opts["token"] = token
    return opts


def get_s3fs_client():
    """Create an s3fs filesystem client."""
    opts = get_s3_storage_options()
    if opts:
        return s3fs.S3FileSystem(key=opts.get("key"), secret=opts.get("secret"),
                                  token=opts.get("token"))
    return s3fs.S3FileSystem()


def load_hourly_counts(storage_options):
    """Load hourly counts from Stage 1."""
    path = f"{S3_INTERMEDIATE}/hourly_counts.parquet"
    logger.info(f"Loading hourly counts from {path}")
    df = pd.read_parquet(path, storage_options=storage_options)
    df["hour_bucket"] = pd.to_datetime(df["hour_bucket"])
    logger.info(f"  Loaded {len(df):,} rows")
    return df


def load_anomaly_windows(storage_options):
    """Load anomaly windows from Stage 2."""
    path = f"{S3_INTERMEDIATE}/anomaly_windows.parquet"
    logger.info(f"Loading anomaly windows from {path}")
    df = pd.read_parquet(path, storage_options=storage_options)
    if "window_start" in df.columns:
        df = df.rename(columns={"window_start": "start_time", "window_end": "end_time"})
    logger.info(f"  Loaded {len(df)} anomaly windows")
    return df


def load_sentiment(storage_options):
    """Load sentiment data from Stage 7."""
    path = f"{S3_INTERMEDIATE}/sentiment.parquet"
    try:
        df = pd.read_parquet(path, storage_options=storage_options)
        logger.info(f"  Loaded sentiment: {len(df)} rows")
        return df
    except Exception:
        logger.warning("  Sentiment data not available")
        return pd.DataFrame()


def compute_baseline(hourly_counts, subreddit):
    """Compute baseline activity for a subreddit (median hourly posts)."""
    sub_data = hourly_counts[hourly_counts["subreddit"] == subreddit]
    if len(sub_data) == 0:
        return 0
    return sub_data["post_count"].median()


def label_sustain_decay(anomaly_windows, hourly_counts):
    """Label each anomaly window as 'sustained' or 'decayed'.

    Sustained: activity stays >2x baseline for >24h after spike start
    Decayed: activity drops below 2x baseline within 24h
    """
    labels = []

    for idx, (_, window) in enumerate(tqdm(anomaly_windows.iterrows(),
                                            total=len(anomaly_windows),
                                            desc="Labeling sustain/decay")):
        subreddit = window["subreddit"]
        start_time = pd.Timestamp(window["start_time"])

        # Compute baseline for this subreddit
        baseline = compute_baseline(hourly_counts, subreddit)
        threshold = baseline * SUSTAIN_THRESHOLD_MULTIPLIER

        if baseline == 0:
            labels.append("decayed")
            continue

        # Look at activity from start to start + 48h
        monitor_end = start_time + timedelta(hours=48)
        sub_hourly = hourly_counts[
            (hourly_counts["subreddit"] == subreddit) &
            (hourly_counts["hour_bucket"] >= start_time) &
            (hourly_counts["hour_bucket"] <= monitor_end)
        ].sort_values("hour_bucket")

        if len(sub_hourly) == 0:
            labels.append("decayed")
            continue

        # Check if activity stays above threshold for 24+ consecutive hours
        above_threshold = sub_hourly["post_count"] > threshold
        max_consecutive = 0
        current_streak = 0

        for val in above_threshold:
            if val:
                current_streak += 1
                max_consecutive = max(max_consecutive, current_streak)
            else:
                current_streak = 0

        if max_consecutive >= SUSTAIN_DURATION_HOURS:
            labels.append("sustained")
        else:
            labels.append("decayed")

    return labels


def extract_early_features(anomaly_windows, hourly_counts, sentiment_df):
    """Extract features from the first 2-4 hours of each spike.

    Features:
    - velocity: rate of post count increase in first N hours
    - author_diversity: unique authors / total posts in early window
    - cross_subreddit_count: number of subreddits with spikes in same period
    - sentiment_intensity: absolute mean sentiment
    - score_acceleration: rate of change in mean score
    - peak_post_count: max hourly posts in early window
    - early_mean_score: mean score in early window
    """
    feature_rows = []

    for idx, (_, window) in enumerate(tqdm(anomaly_windows.iterrows(),
                                            total=len(anomaly_windows),
                                            desc="Extracting early features")):
        subreddit = window["subreddit"]
        start_time = pd.Timestamp(window["start_time"])
        early_end = start_time + timedelta(hours=EARLY_WINDOW_HOURS)

        # Hourly data for this subreddit in early window
        early_data = hourly_counts[
            (hourly_counts["subreddit"] == subreddit) &
            (hourly_counts["hour_bucket"] >= start_time) &
            (hourly_counts["hour_bucket"] <= early_end)
        ].sort_values("hour_bucket")

        # Baseline for context
        baseline = compute_baseline(hourly_counts, subreddit)

        # ── Velocity ────────────────────────────────────────────────────────
        if len(early_data) >= 2:
            counts = early_data["post_count"].values
            velocity = (counts[-1] - counts[0]) / max(len(counts) - 1, 1)
        else:
            velocity = 0

        # ── Author diversity ────────────────────────────────────────────────
        total_posts = early_data["post_count"].sum() if len(early_data) > 0 else 0
        total_authors = early_data["unique_authors"].sum() if len(early_data) > 0 else 0
        author_diversity = total_authors / max(total_posts, 1)

        # ── Cross-subreddit count ───────────────────────────────────────────
        # Count how many other subreddits also have elevated activity at the same time
        concurrent = hourly_counts[
            (hourly_counts["hour_bucket"] >= start_time) &
            (hourly_counts["hour_bucket"] <= early_end) &
            (hourly_counts["subreddit"] != subreddit)
        ]
        if len(concurrent) > 0 and baseline > 0:
            # Subreddits with >2x their own baseline in this period
            concurrent_agg = concurrent.groupby("subreddit")["post_count"].mean()
            # Rough heuristic: subreddits with high activity
            cross_count = (concurrent_agg > concurrent_agg.median() * 2).sum()
        else:
            cross_count = 0

        # ── Score acceleration ──────────────────────────────────────────────
        if len(early_data) >= 2:
            scores = early_data["mean_score"].values
            score_accel = (scores[-1] - scores[0]) / max(len(scores) - 1, 1)
        else:
            score_accel = 0

        # ── Peak and mean ───────────────────────────────────────────────────
        peak_posts = early_data["post_count"].max() if len(early_data) > 0 else 0
        early_mean_score = early_data["mean_score"].mean() if len(early_data) > 0 else 0

        # ── Spike ratio ─────────────────────────────────────────────────────
        spike_ratio = peak_posts / max(baseline, 1)

        # ── Sentiment intensity (from Stage 7 if available) ─────────────────
        sentiment_intensity = 0.0
        window_id = window.get("window_id", idx)
        if len(sentiment_df) > 0 and "window_id" in sentiment_df.columns:
            sent_row = sentiment_df[sentiment_df["window_id"] == window_id]
            if len(sent_row) > 0:
                mean_sent = sent_row["anomaly_mean_sentiment"].iloc[0]
                sentiment_intensity = abs(mean_sent) if not pd.isna(mean_sent) else 0.0

        # ── Time features ───────────────────────────────────────────────────
        hour_of_day = start_time.hour
        is_weekend = int(start_time.dayofweek in [5, 6])

        feature_rows.append({
            "window_id": window_id,
            "subreddit": subreddit,
            "start_time": start_time,
            "velocity": velocity,
            "author_diversity": author_diversity,
            "cross_subreddit_count": cross_count,
            "sentiment_intensity": sentiment_intensity,
            "score_acceleration": score_accel,
            "peak_post_count": peak_posts,
            "early_mean_score": early_mean_score,
            "spike_ratio": spike_ratio,
            "baseline_activity": baseline,
            "hour_of_day": hour_of_day,
            "is_weekend": is_weekend,
        })

    return pd.DataFrame(feature_rows)


def main():
    logger.info("=" * 70)
    logger.info("Stage 10: Spike Sustain vs Decay Prediction")
    logger.info("=" * 70)
    t_start = time.time()

    storage_options = get_s3_storage_options()
    s3 = get_s3fs_client()

    # Check for existing output
    output_path = f"{S3_INTERMEDIATE}/sustain_predictions.parquet"
    try:
        if s3.exists(output_path.replace("s3://", "")):
            logger.info("sustain_predictions.parquet already exists. Skipping.")
            return
    except Exception:
        pass

    # ── Load data ───────────────────────────────────────────────────────────
    hourly_counts = load_hourly_counts(storage_options)
    anomaly_windows = load_anomaly_windows(storage_options)
    sentiment_df = load_sentiment(storage_options)

    if len(anomaly_windows) == 0:
        logger.error("No anomaly windows. Run prior stages first.")
        sys.exit(1)

    # ── Label sustain vs decay ──────────────────────────────────────────────
    logger.info("Labeling anomaly windows as sustained or decayed...")
    labels = label_sustain_decay(anomaly_windows, hourly_counts)
    anomaly_windows["sustain_label"] = labels

    n_sustained = sum(1 for l in labels if l == "sustained")
    n_decayed = sum(1 for l in labels if l == "decayed")
    logger.info(f"  Sustained: {n_sustained}, Decayed: {n_decayed}")

    if n_sustained < 3 or n_decayed < 3:
        logger.warning("Very few examples in one class. Results may be unreliable.")

    # ── Extract early-window features ───────────────────────────────────────
    logger.info("Extracting features from first 4 hours of each spike...")
    features_df = extract_early_features(
        anomaly_windows, hourly_counts, sentiment_df
    )

    # Merge labels
    if "window_id" in anomaly_windows.columns:
        features_df = features_df.merge(
            anomaly_windows[["window_id", "sustain_label"]],
            on="window_id", how="left",
        )
    else:
        features_df["sustain_label"] = labels

    # ── Prepare feature arrays ──────────────────────────────────────────────
    feature_cols = [
        "velocity", "author_diversity", "cross_subreddit_count",
        "sentiment_intensity", "score_acceleration", "peak_post_count",
        "early_mean_score", "spike_ratio", "baseline_activity",
        "hour_of_day", "is_weekend",
    ]

    X = features_df[feature_cols].values.astype(np.float32)
    y = (features_df["sustain_label"] == "sustained").astype(int).values

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Class balance: sustained={y.sum()}, decayed={len(y) - y.sum()}")

    # ── Time-based train/test split ─────────────────────────────────────────
    features_df["start_time_ts"] = pd.to_datetime(features_df["start_time"])
    split_date = features_df["start_time_ts"].quantile(0.75)
    logger.info(f"Time-based split at: {split_date}")

    train_mask = features_df["start_time_ts"] <= split_date
    test_mask = features_df["start_time_ts"] > split_date

    X_train, y_train = X[train_mask.values], y[train_mask.values]
    X_test, y_test = X[test_mask.values], y[test_mask.values]

    logger.info(f"Train set: {len(X_train)} (sustained={y_train.sum()}, "
                f"decayed={len(y_train) - y_train.sum()})")
    logger.info(f"Test set:  {len(X_test)} (sustained={y_test.sum()}, "
                f"decayed={len(y_test) - y_test.sum()})")

    if len(X_test) == 0 or len(set(y_test)) < 2:
        logger.warning("Test set is empty or has only one class. "
                        "Falling back to random 80/20 split.")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None,
        )

    # ── Scale features ──────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Train RandomForest ──────────────────────────────────────────────────
    logger.info("Training RandomForest classifier...")

    if GPU_ML:
        model = cuRF(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_streams=1,
        )
        model.fit(
            cudf.DataFrame(X_train_scaled, columns=feature_cols),
            cudf.Series(y_train),
        )
        y_pred = model.predict(cudf.DataFrame(X_test_scaled, columns=feature_cols))
        if hasattr(y_pred, "to_pandas"):
            y_pred = y_pred.to_pandas().values
        try:
            y_prob = model.predict_proba(cudf.DataFrame(X_test_scaled, columns=feature_cols))
            if hasattr(y_prob, "to_pandas"):
                y_prob = y_prob.to_pandas().values
            y_prob = y_prob[:, 1]
        except Exception:
            y_prob = y_pred.astype(float)

        importances = model.feature_importances_
        if hasattr(importances, "values_host"):
            importances = importances.values_host
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        importances = model.feature_importances_

    # ── Evaluate ────────────────────────────────────────────────────────────
    if len(set(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = float("nan")
        logger.warning("Cannot compute ROC-AUC: only one class in test set")

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info(f"\nModel Evaluation:")
    logger.info(f"  ROC-AUC:    {roc_auc:.3f}")
    logger.info(f"  Precision:  {precision:.3f}")
    logger.info(f"  Recall:     {recall:.3f}")
    logger.info(f"  F1 Score:   {f1:.3f}")
    logger.info(f"\nClassification Report:\n"
                f"{classification_report(y_test, y_pred, target_names=['decayed', 'sustained'], zero_division=0)}")

    # Feature importance
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    logger.info("\nFeature Importance:")
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")

    # ── Predict on all windows ──────────────────────────────────────────────
    X_all_scaled = scaler.transform(X)
    if GPU_ML:
        all_preds = model.predict(cudf.DataFrame(X_all_scaled, columns=feature_cols))
        if hasattr(all_preds, "to_pandas"):
            all_preds = all_preds.to_pandas().values
        try:
            all_probs = model.predict_proba(cudf.DataFrame(X_all_scaled, columns=feature_cols))
            if hasattr(all_probs, "to_pandas"):
                all_probs = all_probs.to_pandas().values
            all_probs = all_probs[:, 1]
        except Exception:
            all_probs = all_preds.astype(float)
    else:
        all_preds = model.predict(X_all_scaled)
        all_probs = model.predict_proba(X_all_scaled)[:, 1]

    features_df["predicted_sustained"] = all_preds
    features_df["sustain_probability"] = all_probs
    features_df["predicted_label"] = np.where(all_preds == 1, "sustained", "decayed")

    # Add evaluation metrics
    features_df["model_roc_auc"] = roc_auc
    features_df["model_precision"] = precision
    features_df["model_recall"] = recall
    features_df["model_f1"] = f1

    # ── Write output ────────────────────────────────────────────────────────
    logger.info(f"\nWriting sustain_predictions.parquet ({len(features_df)} rows)...")
    features_df.to_parquet(output_path, index=False, storage_options=storage_options)

    # Also write feature importance
    imp_path = f"{S3_INTERMEDIATE}/sustain_feature_importance.parquet"
    importance_df.to_parquet(imp_path, index=False, storage_options=storage_options)

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("Stage 10 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time:           {elapsed / 60:.1f} minutes")
    logger.info(f"  Total windows:        {len(features_df)}")
    logger.info(f"  Sustained:            {(features_df['sustain_label'] == 'sustained').sum()}")
    logger.info(f"  Decayed:              {(features_df['sustain_label'] == 'decayed').sum()}")
    logger.info(f"  ROC-AUC:              {roc_auc:.3f}")
    logger.info(f"  Precision:            {precision:.3f}")
    logger.info(f"  Recall:               {recall:.3f}")
    logger.info(f"  F1 Score:             {f1:.3f}")
    logger.info(f"  Top feature:          {importance_df.iloc[0]['feature']}")
    logger.info(f"  Output at:            {output_path}")


if __name__ == "__main__":
    main()
