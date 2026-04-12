#!/usr/bin/env python3
"""
Stage 11: Event Forecasting - Bonus (Q10)
============================================
Predicts whether a major event will occur in the next 24 hours based on
precursor signals from the preceding 48-hour window.

Uses ground truth events as positive examples and random non-event windows
as negative examples. Reports honest precision/recall (expects modest results).

Outputs:
  - forecast_results.parquet: predictions with evaluation metrics

Usage: python stage11_forecast_gpu.py
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
    classification_report, precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# ── GPU imports with fallback ───────────────────────────────────────────────
try:
    import cudf
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    GPU_ML = True
    print("cuML available - using GPU RandomForest")
except ImportError:
    GPU_ML = False
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    print("cuML not available - using sklearn")

# Allow CPU-only mode when GPU is shared with NLP stages (run_all_gpu.sh)
if os.environ.get("FORCE_CPU", "0") == "1":
    GPU_ML = False
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    print("FORCE_CPU=1: Using sklearn on EPYC cores to avoid GPU contention")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ── Logging ─────────────────────────────────────────────────────────────────
os.makedirs("/workspace/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/workspace/logs/stage11_forecast.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
S3_BUCKET = "ven-bda-s3-v2"
S3_INTERMEDIATE = f"s3://{S3_BUCKET}/reddit-data/intermediate"

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUND_TRUTH_PATH = os.path.join(PROJECT_DIR, "data", "ground_truth", "events.csv")

PRE_EVENT_WINDOW_HOURS = 48  # look at 48h before event
NEGATIVE_RATIO = 3  # 3 negative examples per positive


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


def load_ground_truth():
    """Load ground truth events."""
    logger.info(f"Loading ground truth from {GROUND_TRUTH_PATH}")
    gt = pd.read_csv(GROUND_TRUTH_PATH)
    gt["date"] = pd.to_datetime(gt["date"])
    gt["relevant_subreddits"] = gt["relevant_subreddits"].str.split("|")
    logger.info(f"  Loaded {len(gt)} events")
    return gt


def load_hourly_counts(storage_options):
    """Load hourly counts from Stage 1."""
    path = f"{S3_INTERMEDIATE}/hourly_counts.parquet"
    logger.info(f"Loading hourly counts from {path}")
    df = pd.read_parquet(path, storage_options=storage_options)
    df["hour_bucket"] = pd.to_datetime(df["hour_bucket"])
    logger.info(f"  Loaded {len(df):,} rows")
    return df


def load_sentiment(storage_options):
    """Load sentiment data from Stage 7 if available."""
    path = f"{S3_INTERMEDIATE}/sentiment.parquet"
    try:
        df = pd.read_parquet(path, storage_options=storage_options)
        return df
    except Exception:
        return pd.DataFrame()


def compute_subreddit_baselines(hourly_counts):
    """Compute baseline statistics for each subreddit."""
    baselines = hourly_counts.groupby("subreddit").agg(
        median_posts=("post_count", "median"),
        mean_posts=("post_count", "mean"),
        std_posts=("post_count", "std"),
        median_authors=("unique_authors", "median"),
        median_score=("mean_score", "median"),
    ).reset_index()
    baselines["std_posts"] = baselines["std_posts"].fillna(1)
    return baselines


def extract_precursor_features(hourly_counts, subreddits, event_time,
                                baselines, window_hours=PRE_EVENT_WINDOW_HOURS):
    """Extract precursor features from the 48h window before an event.

    Features:
    - activity_trend: slope of hourly post count over window
    - cross_posting_rate: number of subreddits with rising activity
    - sentiment_drift: change in activity pattern
    - author_concentration: Gini coefficient of author distribution
    - mean_z_score: average z-score across subreddits in window
    - max_z_score: peak z-score
    - activity_acceleration: second derivative of post count
    - volume_ratio: activity in last 12h / first 12h of window
    - score_trend: slope of mean score
    - author_growth: trend in unique authors
    """
    window_start = event_time - timedelta(hours=window_hours)
    window_end = event_time

    # Get data for relevant subreddits in the window
    features = {}
    all_activity = []
    all_authors = []
    all_scores = []

    for sub in subreddits:
        sub_data = hourly_counts[
            (hourly_counts["subreddit"] == sub) &
            (hourly_counts["hour_bucket"] >= window_start) &
            (hourly_counts["hour_bucket"] < window_end)
        ].sort_values("hour_bucket")

        if len(sub_data) == 0:
            continue

        all_activity.extend(sub_data["post_count"].tolist())
        all_authors.extend(sub_data["unique_authors"].tolist())
        all_scores.extend(sub_data["mean_score"].tolist())

        # Get baseline for z-score computation
        baseline_row = baselines[baselines["subreddit"] == sub]
        if len(baseline_row) > 0:
            bl_mean = baseline_row["mean_posts"].iloc[0]
            bl_std = max(baseline_row["std_posts"].iloc[0], 1)
            sub_data_z = (sub_data["post_count"] - bl_mean) / bl_std
        else:
            sub_data_z = sub_data["post_count"]  # fallback

    if not all_activity:
        return None

    activity = np.array(all_activity, dtype=float)
    authors = np.array(all_authors, dtype=float)
    scores = np.array(all_scores, dtype=float)

    # ── Activity trend (linear regression slope) ────────────────────────────
    if len(activity) > 2:
        x = np.arange(len(activity))
        activity_trend = np.polyfit(x, activity, 1)[0]
    else:
        activity_trend = 0

    # ── Cross-posting rate ──────────────────────────────────────────────────
    # How many subreddits show above-baseline activity
    cross_count = 0
    for sub in subreddits:
        sub_recent = hourly_counts[
            (hourly_counts["subreddit"] == sub) &
            (hourly_counts["hour_bucket"] >= window_end - timedelta(hours=12)) &
            (hourly_counts["hour_bucket"] < window_end)
        ]
        bl_row = baselines[baselines["subreddit"] == sub]
        if len(sub_recent) > 0 and len(bl_row) > 0:
            if sub_recent["post_count"].mean() > bl_row["mean_posts"].iloc[0] * 1.5:
                cross_count += 1
    cross_posting_rate = cross_count / max(len(subreddits), 1)

    # ── Author concentration (Gini coefficient) ────────────────────────────
    if len(authors) > 1:
        sorted_a = np.sort(authors)
        n = len(sorted_a)
        cumsum = np.cumsum(sorted_a)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_a)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1]) if cumsum[-1] > 0 else 0
        author_concentration = max(0, min(gini, 1))
    else:
        author_concentration = 0

    # ── Z-score statistics ──────────────────────────────────────────────────
    if len(activity) > 0 and np.std(activity) > 0:
        z_scores = (activity - np.mean(activity)) / max(np.std(activity), 1)
        mean_z = float(np.mean(z_scores))
        max_z = float(np.max(z_scores))
    else:
        mean_z = 0
        max_z = 0

    # ── Activity acceleration ───────────────────────────────────────────────
    if len(activity) > 3:
        first_diff = np.diff(activity)
        second_diff = np.diff(first_diff)
        activity_acceleration = float(np.mean(second_diff))
    else:
        activity_acceleration = 0

    # ── Volume ratio (last 12h vs first 12h) ────────────────────────────────
    mid = len(activity) // 2
    if mid > 0:
        first_half = np.mean(activity[:mid]) if mid > 0 else 1
        second_half = np.mean(activity[mid:])
        volume_ratio = second_half / max(first_half, 1)
    else:
        volume_ratio = 1

    # ── Score trend ─────────────────────────────────────────────────────────
    if len(scores) > 2:
        score_trend = np.polyfit(np.arange(len(scores)), scores, 1)[0]
    else:
        score_trend = 0

    # ── Author growth ───────────────────────────────────────────────────────
    if len(authors) > 2:
        author_growth = np.polyfit(np.arange(len(authors)), authors, 1)[0]
    else:
        author_growth = 0

    features = {
        "activity_trend": activity_trend,
        "cross_posting_rate": cross_posting_rate,
        "author_concentration": author_concentration,
        "mean_z_score": mean_z,
        "max_z_score": max_z,
        "activity_acceleration": activity_acceleration,
        "volume_ratio": volume_ratio,
        "score_trend": score_trend,
        "author_growth": author_growth,
        "mean_activity": float(np.mean(activity)),
        "std_activity": float(np.std(activity)),
        "peak_activity": float(np.max(activity)),
        "mean_authors": float(np.mean(authors)),
        "mean_score": float(np.mean(scores)) if len(scores) > 0 else 0,
        "n_hours_data": len(activity),
    }

    return features


def create_negative_examples(hourly_counts, ground_truth, baselines,
                              n_negatives_per_positive=NEGATIVE_RATIO):
    """Create negative examples from random non-event windows.

    Ensures no overlap with any ground truth event +/- 3 days.
    """
    # Get all event dates with buffer
    event_dates = set()
    for _, event in ground_truth.iterrows():
        for d in range(-3, 4):
            event_dates.add((event["date"] + timedelta(days=d)).date())

    # Get all available dates
    all_dates = hourly_counts["hour_bucket"].dt.date.unique()
    non_event_dates = [d for d in all_dates if d not in event_dates]

    if len(non_event_dates) == 0:
        logger.warning("No non-event dates available for negative examples")
        return []

    # Get top subreddits for sampling
    top_subs = (hourly_counts.groupby("subreddit")["post_count"].sum()
                .sort_values(ascending=False).head(50).index.tolist())

    n_needed = len(ground_truth) * n_negatives_per_positive
    rng = np.random.RandomState(42)

    negatives = []
    attempts = 0
    max_attempts = n_needed * 10

    while len(negatives) < n_needed and attempts < max_attempts:
        attempts += 1
        rand_date = pd.Timestamp(rng.choice(non_event_dates))
        rand_hour = rng.randint(0, 24)
        rand_time = rand_date + timedelta(hours=rand_hour)
        rand_sub = rng.choice(top_subs)

        features = extract_precursor_features(
            hourly_counts, [rand_sub], rand_time, baselines
        )

        if features is not None:
            features["label"] = 0
            features["event_name"] = "non_event"
            features["event_time"] = rand_time
            features["subreddits"] = rand_sub
            negatives.append(features)

    logger.info(f"  Created {len(negatives)} negative examples from "
                f"{attempts} attempts")
    return negatives


def main():
    logger.info("=" * 70)
    logger.info("Stage 11: Event Forecasting (Bonus)")
    logger.info("=" * 70)
    t_start = time.time()

    storage_options = get_s3_storage_options()
    s3 = get_s3fs_client()

    # Check for existing output
    output_path = f"{S3_INTERMEDIATE}/forecast_results.parquet"
    try:
        if s3.exists(output_path.replace("s3://", "")):
            logger.info("forecast_results.parquet already exists. Skipping.")
            return
    except Exception:
        pass

    # ── Load data ───────────────────────────────────────────────────────────
    ground_truth = load_ground_truth()
    hourly_counts = load_hourly_counts(storage_options)

    # Compute baselines
    logger.info("Computing subreddit baselines...")
    baselines = compute_subreddit_baselines(hourly_counts)

    # ── Extract positive examples (48h pre-event windows) ───────────────────
    logger.info("Extracting precursor features for ground truth events...")
    positive_examples = []

    for _, event in tqdm(ground_truth.iterrows(), total=len(ground_truth),
                          desc="Positive examples"):
        event_time = event["date"]
        subreddits = event["relevant_subreddits"]

        features = extract_precursor_features(
            hourly_counts, subreddits, event_time, baselines
        )

        if features is not None:
            features["label"] = 1
            features["event_name"] = event["name"]
            features["event_time"] = event_time
            features["event_category"] = event["category"]
            features["subreddits"] = "|".join(subreddits)
            positive_examples.append(features)

    logger.info(f"  Extracted {len(positive_examples)} positive examples "
                f"from {len(ground_truth)} events")

    if len(positive_examples) < 5:
        logger.error("Too few positive examples. Cannot train forecast model.")
        sys.exit(1)

    # ── Create negative examples ────────────────────────────────────────────
    logger.info("Creating negative (non-event) examples...")
    negative_examples = create_negative_examples(
        hourly_counts, ground_truth, baselines
    )

    # ── Combine and prepare dataset ─────────────────────────────────────────
    all_examples = positive_examples + negative_examples
    dataset = pd.DataFrame(all_examples)

    logger.info(f"Total dataset: {len(dataset)} samples "
                f"(positive={len(positive_examples)}, negative={len(negative_examples)})")

    feature_cols = [
        "activity_trend", "cross_posting_rate", "author_concentration",
        "mean_z_score", "max_z_score", "activity_acceleration",
        "volume_ratio", "score_trend", "author_growth",
        "mean_activity", "std_activity", "peak_activity",
        "mean_authors", "mean_score", "n_hours_data",
    ]

    X = dataset[feature_cols].values.astype(np.float32)
    y = dataset["label"].values.astype(int)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Stratified K-Fold evaluation ────────────────────────────────────────
    logger.info("Running 5-fold stratified cross-validation...")
    n_folds = min(5, len(positive_examples))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []
    all_fold_preds = np.zeros(len(y))
    all_fold_probs = np.zeros(len(y))

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if GPU_ML:
            model = cuRF(
                n_estimators=200,
                max_depth=8,
                random_state=42,
                n_streams=1,
            )
            model.fit(
                cudf.DataFrame(X_train_s, columns=feature_cols),
                cudf.Series(y_train),
            )
            preds = model.predict(cudf.DataFrame(X_test_s, columns=feature_cols))
            if hasattr(preds, "to_pandas"):
                preds = preds.to_pandas().values
            try:
                probs = model.predict_proba(cudf.DataFrame(X_test_s, columns=feature_cols))
                if hasattr(probs, "to_pandas"):
                    probs = probs.to_pandas().values
                probs = probs[:, 1]
            except Exception:
                probs = preds.astype(float)
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                random_state=42,
                class_weight="balanced",
            )
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            probs = model.predict_proba(X_test_s)[:, 1]

        all_fold_preds[test_idx] = preds
        all_fold_probs[test_idx] = probs

        # Fold metrics
        if len(set(y_test)) > 1:
            fold_auc = roc_auc_score(y_test, probs)
        else:
            fold_auc = float("nan")

        fold_prec = precision_score(y_test, preds, zero_division=0)
        fold_rec = recall_score(y_test, preds, zero_division=0)
        fold_f1 = f1_score(y_test, preds, zero_division=0)

        fold_metrics.append({
            "fold": fold,
            "auc": fold_auc,
            "precision": fold_prec,
            "recall": fold_rec,
            "f1": fold_f1,
        })

        logger.info(f"  Fold {fold}: AUC={fold_auc:.3f} P={fold_prec:.3f} "
                    f"R={fold_rec:.3f} F1={fold_f1:.3f}")

    # ── Overall metrics ─────────────────────────────────────────────────────
    if len(set(y)) > 1:
        overall_auc = roc_auc_score(y, all_fold_probs)
        overall_ap = average_precision_score(y, all_fold_probs)
    else:
        overall_auc = float("nan")
        overall_ap = float("nan")

    overall_prec = precision_score(y, all_fold_preds, zero_division=0)
    overall_rec = recall_score(y, all_fold_preds, zero_division=0)
    overall_f1 = f1_score(y, all_fold_preds, zero_division=0)

    logger.info(f"\nOverall Cross-Validation Results:")
    logger.info(f"  ROC-AUC:            {overall_auc:.3f}")
    logger.info(f"  Average Precision:  {overall_ap:.3f}")
    logger.info(f"  Precision:          {overall_prec:.3f}")
    logger.info(f"  Recall:             {overall_rec:.3f}")
    logger.info(f"  F1 Score:           {overall_f1:.3f}")

    logger.info(f"\nClassification Report:\n"
                f"{classification_report(y, all_fold_preds, target_names=['no_event', 'event'], zero_division=0)}")

    # ── Train final model on all data for feature importance ────────────────
    logger.info("Training final model on all data for feature importance...")
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)

    if GPU_ML:
        final_model = cuRF(n_estimators=300, max_depth=10, random_state=42, n_streams=1)
        final_model.fit(
            cudf.DataFrame(X_scaled, columns=feature_cols),
            cudf.Series(y),
        )
        importances = final_model.feature_importances_
        if hasattr(importances, "values_host"):
            importances = importances.values_host
    else:
        final_model = RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=42, class_weight="balanced"
        )
        final_model.fit(X_scaled, y)
        importances = final_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    logger.info("\nFeature Importance for Forecasting:")
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")

    # ── Compile output ──────────────────────────────────────────────────────
    dataset["predicted_event"] = all_fold_preds.astype(int)
    dataset["event_probability"] = all_fold_probs

    # Add metrics
    dataset["model_roc_auc"] = overall_auc
    dataset["model_avg_precision"] = overall_ap
    dataset["model_precision"] = overall_prec
    dataset["model_recall"] = overall_rec
    dataset["model_f1"] = overall_f1

    # ── Write outputs ───────────────────────────────────────────────────────
    logger.info(f"\nWriting forecast_results.parquet ({len(dataset)} rows)...")
    dataset.to_parquet(output_path, index=False, storage_options=storage_options)

    imp_path = f"{S3_INTERMEDIATE}/forecast_feature_importance.parquet"
    importance_df.to_parquet(imp_path, index=False, storage_options=storage_options)

    fold_metrics_path = f"{S3_INTERMEDIATE}/forecast_fold_metrics.parquet"
    pd.DataFrame(fold_metrics).to_parquet(
        fold_metrics_path, index=False, storage_options=storage_options
    )

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("Stage 11 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time:           {elapsed / 60:.1f} minutes")
    logger.info(f"  Positive examples:    {len(positive_examples)}")
    logger.info(f"  Negative examples:    {len(negative_examples)}")
    logger.info(f"  ROC-AUC:              {overall_auc:.3f}")
    logger.info(f"  Average Precision:    {overall_ap:.3f}")
    logger.info(f"  Precision:            {overall_prec:.3f}")
    logger.info(f"  Recall:               {overall_rec:.3f}")
    logger.info(f"  F1 Score:             {overall_f1:.3f}")
    logger.info(f"")
    logger.info(f"  NOTE: Forecasting real-world events from Reddit precursor")
    logger.info(f"  signals is inherently difficult. Modest metrics are expected.")
    logger.info(f"  The value is in identifying which precursor features have")
    logger.info(f"  any predictive signal at all.")
    logger.info(f"")
    logger.info(f"  Top precursor features:")
    for _, row in importance_df.head(5).iterrows():
        logger.info(f"    {row['feature']:30s} {row['importance']:.4f}")
    logger.info(f"  Output at:            {output_path}")


if __name__ == "__main__":
    main()
