#!/usr/bin/env python3
"""
Stage 9: GPU-Accelerated Event Classification (Q8)
=====================================================
Joins features from prior stages with ground truth events to build a
5-class event classifier: breaking_news, controversy, product_launch,
disaster, meme_viral.

Uses cuML RandomForest and XGBoost with LOOCV evaluation.

Outputs:
  - classifications.parquet: predicted event categories per anomaly window
  - feature_importance.parquet: feature importance rankings

Usage: python stage9_classification_gpu.py
"""

import os
import sys
import time
import logging
import gc
import json
from datetime import datetime

import numpy as np
import pandas as pd
import s3fs
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

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

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("XGBoost available")
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available")

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("stage9_classification.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
S3_BUCKET = "ven-bda-s3-v2"
S3_INTERMEDIATE = f"s3://{S3_BUCKET}/reddit-data/intermediate"

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUND_TRUTH_PATH = os.path.join(PROJECT_DIR, "data", "ground_truth", "events.csv")

EVENT_CATEGORIES = ["breaking_news", "controversy", "product_launch", "disaster", "meme_viral"]

FEATURE_COLUMNS = [
    "z_score", "duration_hours", "sentiment_shift", "entity_count",
    "propagation_speed", "num_subreddits", "hour_of_day", "is_weekend",
    "dominant_topic", "mean_score", "unique_authors", "post_count",
    "anomaly_mean_sentiment", "anomaly_std_sentiment",
    "prop_positive", "prop_negative",
]


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
    """Load ground truth events CSV."""
    logger.info(f"Loading ground truth from {GROUND_TRUTH_PATH}")
    gt = pd.read_csv(GROUND_TRUTH_PATH)
    gt["date"] = pd.to_datetime(gt["date"])
    gt["relevant_subreddits"] = gt["relevant_subreddits"].str.split("|")
    logger.info(f"  Loaded {len(gt)} ground truth events")
    logger.info(f"  Categories: {gt['category'].value_counts().to_dict()}")
    return gt


def load_intermediate_data(storage_options):
    """Load all intermediate data from prior stages."""
    data = {}

    files = {
        "anomaly_windows": "anomaly_windows.parquet",
        "entities": "entities.parquet",
        "sentiment": "sentiment.parquet",
        "topics": "topics.parquet",
    }

    for name, filename in files.items():
        path = f"{S3_INTERMEDIATE}/{filename}"
        try:
            df = pd.read_parquet(path, storage_options=storage_options)
            if name == "anomaly_windows" and "window_start" in df.columns:
                df = df.rename(columns={"window_start": "start_time", "window_end": "end_time"})
            data[name] = df
            logger.info(f"  Loaded {name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"  Could not load {name}: {e}")
            data[name] = pd.DataFrame()

    return data


def match_anomalies_to_events(anomaly_windows, ground_truth):
    """Match anomaly windows to ground truth events based on time and subreddit overlap.

    An anomaly window matches an event if:
    1. The anomaly's time range overlaps with +/- 1 day of the event date
    2. The anomaly's subreddit is in the event's relevant_subreddits
    """
    matches = []

    for _, event in ground_truth.iterrows():
        event_date = event["date"]
        event_subs = set(event["relevant_subreddits"])
        event_start = event_date - pd.Timedelta(days=1)
        event_end = event_date + pd.Timedelta(days=1)

        for _, window in anomaly_windows.iterrows():
            window_id = window.get("window_id", window.name)
            subreddit = window["subreddit"]

            # Check subreddit match
            if subreddit not in event_subs:
                continue

            # Check time overlap
            w_start = pd.Timestamp(window["start_time"])
            w_end = pd.Timestamp(window["end_time"])

            if w_start <= event_end and w_end >= event_start:
                matches.append({
                    "window_id": window_id,
                    "event_id": event["event_id"],
                    "event_name": event["name"],
                    "category": event["category"],
                    "subreddit": subreddit,
                })

    matches_df = pd.DataFrame(matches)
    if len(matches_df) > 0:
        # Deduplicate: one window -> one event (closest by date)
        matches_df = matches_df.drop_duplicates(subset=["window_id"])

    logger.info(f"  Matched {len(matches_df)} anomaly windows to ground truth events")
    return matches_df


def build_feature_matrix(anomaly_windows, entities, sentiment, topics, matches):
    """Build feature matrix from all intermediate data."""
    logger.info("Building feature matrix...")

    # Start with anomaly windows
    features = anomaly_windows.copy()

    # Ensure window_id exists
    if "window_id" not in features.columns:
        features["window_id"] = range(len(features))

    # ── Time features ───────────────────────────────────────────────────────
    features["start_dt"] = pd.to_datetime(features["start_time"])
    features["hour_of_day"] = features["start_dt"].dt.hour
    features["is_weekend"] = features["start_dt"].dt.dayofweek.isin([5, 6]).astype(int)

    if "end_time" in features.columns:
        features["end_dt"] = pd.to_datetime(features["end_time"])
        features["duration_hours"] = (
            (features["end_dt"] - features["start_dt"]).dt.total_seconds() / 3600
        )
    else:
        features["duration_hours"] = 0

    # ── Entity features ─────────────────────────────────────────────────────
    if len(entities) > 0:
        ent_agg = entities.groupby("window_id").agg(
            entity_count=("count", "sum"),
            unique_entities=("entity_text", "nunique"),
            person_count=("count", lambda x: x[entities.loc[x.index, "entity_label"] == "PERSON"].sum()),
            org_count=("count", lambda x: x[entities.loc[x.index, "entity_label"] == "ORG"].sum()),
        ).reset_index()
        features = features.merge(ent_agg, on="window_id", how="left")
    else:
        features["entity_count"] = 0
        features["unique_entities"] = 0

    # ── Sentiment features ──────────────────────────────────────────────────
    if len(sentiment) > 0:
        sent_cols = [c for c in sentiment.columns if c in [
            "window_id", "anomaly_mean_sentiment", "anomaly_std_sentiment",
            "anomaly_prop_positive", "anomaly_prop_negative",
            "sentiment_shift", "shift_pvalue", "shift_significant",
        ]]
        if "window_id" in sent_cols:
            features = features.merge(
                sentiment[sent_cols], on="window_id", how="left"
            )

    # ── Topic features ──────────────────────────────────────────────────────
    if len(topics) > 0:
        topic_cols = [c for c in topics.columns if c in [
            "window_id", "dominant_topic", "n_topics_present",
        ]]
        if "window_id" in topic_cols:
            features = features.merge(
                topics[topic_cols], on="window_id", how="left"
            )

    # ── Add labels from matches ─────────────────────────────────────────────
    if len(matches) > 0:
        features = features.merge(
            matches[["window_id", "category", "event_id", "event_name"]],
            on="window_id",
            how="left",
        )

    # Fill NaN for numeric features
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0)

    # Rename columns for consistency
    rename_map = {
        "anomaly_prop_positive": "prop_positive",
        "anomaly_prop_negative": "prop_negative",
    }
    features.rename(columns={k: v for k, v in rename_map.items()
                             if k in features.columns}, inplace=True)

    # Ensure all expected feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in features.columns:
            features[col] = 0

    logger.info(f"  Feature matrix: {features.shape}")
    return features


def loocv_evaluate(X, y, label_encoder, model_type="rf"):
    """Leave-One-Out Cross-Validation for small datasets."""
    n = len(X)
    predictions = []
    true_labels = []

    logger.info(f"Running LOOCV with {n} samples ({model_type})...")

    for i in tqdm(range(n), desc=f"LOOCV ({model_type})"):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i + 1]
        y_test = y[i]

        if model_type == "rf":
            if GPU_ML:
                model = cuRF(
                    n_estimators=200,
                    max_depth=8,
                    random_state=42,
                    n_streams=1,
                )
                model.fit(cudf.DataFrame(X_train), cudf.Series(y_train))
                pred = model.predict(cudf.DataFrame(X_test))
                pred = int(pred.values_host[0]) if hasattr(pred, "values_host") else int(pred.iloc[0])
            else:
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    random_state=42,
                    class_weight="balanced",
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_test)[0]

        elif model_type == "xgb" and XGB_AVAILABLE:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test)
            params = {
                "max_depth": 6,
                "eta": 0.1,
                "objective": "multi:softmax",
                "num_class": len(label_encoder.classes_),
                "eval_metric": "mlogloss",
                "tree_method": "gpu_hist" if GPU_ML else "hist",
                "seed": 42,
            }
            bst = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
            pred = int(bst.predict(dtest)[0])
        else:
            continue

        predictions.append(pred)
        true_labels.append(y_test)

    return np.array(predictions), np.array(true_labels)


def train_full_model(X, y, feature_names, label_encoder):
    """Train on full labeled data and return feature importance."""
    logger.info("Training full model on all labeled data...")

    if GPU_ML:
        model = cuRF(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_streams=1,
        )
        model.fit(cudf.DataFrame(X), cudf.Series(y))
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
        model.fit(X, y)
        importances = model.feature_importances_

    # Feature importance DataFrame
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return model, importance_df


def main():
    logger.info("=" * 70)
    logger.info("Stage 9: GPU-Accelerated Event Classification")
    logger.info("=" * 70)
    t_start = time.time()

    storage_options = get_s3_storage_options()
    s3 = get_s3fs_client()

    # Check for existing output
    output_path = f"{S3_INTERMEDIATE}/classifications.parquet"
    importance_path = f"{S3_INTERMEDIATE}/feature_importance.parquet"
    try:
        if s3.exists(output_path.replace("s3://", "")):
            logger.info("classifications.parquet already exists on S3. Skipping.")
            return
    except Exception:
        pass

    # ── Load all data ───────────────────────────────────────────────────────
    logger.info("Loading intermediate data...")
    ground_truth = load_ground_truth()
    data = load_intermediate_data(storage_options)

    anomaly_windows = data["anomaly_windows"]
    if len(anomaly_windows) == 0:
        logger.error("No anomaly windows found. Run prior stages first.")
        sys.exit(1)

    # ── Match anomaly windows to events ─────────────────────────────────────
    matches = match_anomalies_to_events(anomaly_windows, ground_truth)

    # ── Build feature matrix ────────────────────────────────────────────────
    features = build_feature_matrix(
        anomaly_windows, data["entities"], data["sentiment"],
        data["topics"], matches,
    )

    # ── Separate labeled vs unlabeled ───────────────────────────────────────
    labeled = features[features["category"].notna()].copy()
    unlabeled = features[features["category"].isna()].copy()

    logger.info(f"Labeled windows:   {len(labeled)}")
    logger.info(f"Unlabeled windows: {len(unlabeled)}")
    logger.info(f"Category distribution:\n{labeled['category'].value_counts().to_string()}")

    if len(labeled) < 5:
        logger.error("Too few labeled samples for classification. "
                      "Need at least 5 matched events.")
        # Still write out features for analysis
        features.to_parquet(output_path, index=False, storage_options=storage_options)
        logger.info(f"Feature matrix saved to {output_path} (no classification performed)")
        return

    # ── Prepare arrays ──────────────────────────────────────────────────────
    available_features = [c for c in FEATURE_COLUMNS if c in labeled.columns]
    logger.info(f"Using features: {available_features}")

    X_labeled = labeled[available_features].values.astype(np.float32)
    label_encoder = LabelEncoder()
    y_labeled = label_encoder.fit_transform(labeled["category"].values)

    logger.info(f"Classes: {list(label_encoder.classes_)}")
    logger.info(f"Feature matrix shape: {X_labeled.shape}")

    # ── LOOCV Evaluation: RandomForest ──────────────────────────────────────
    rf_preds, rf_true = loocv_evaluate(X_labeled, y_labeled, label_encoder, "rf")

    rf_accuracy = accuracy_score(rf_true, rf_preds)
    rf_f1_macro = f1_score(rf_true, rf_preds, average="macro", zero_division=0)
    rf_f1_weighted = f1_score(rf_true, rf_preds, average="weighted", zero_division=0)

    logger.info(f"\nRandomForest LOOCV Results:")
    logger.info(f"  Accuracy:         {rf_accuracy:.3f}")
    logger.info(f"  F1 (macro):       {rf_f1_macro:.3f}")
    logger.info(f"  F1 (weighted):    {rf_f1_weighted:.3f}")
    logger.info(f"\nClassification Report:\n"
                f"{classification_report(rf_true, rf_preds, target_names=label_encoder.classes_, zero_division=0)}")

    # ── LOOCV Evaluation: XGBoost ───────────────────────────────────────────
    xgb_results = {}
    if XGB_AVAILABLE:
        xgb_preds, xgb_true = loocv_evaluate(X_labeled, y_labeled, label_encoder, "xgb")

        xgb_accuracy = accuracy_score(xgb_true, xgb_preds)
        xgb_f1_macro = f1_score(xgb_true, xgb_preds, average="macro", zero_division=0)

        logger.info(f"\nXGBoost LOOCV Results:")
        logger.info(f"  Accuracy:         {xgb_accuracy:.3f}")
        logger.info(f"  F1 (macro):       {xgb_f1_macro:.3f}")
        logger.info(f"\nClassification Report:\n"
                    f"{classification_report(xgb_true, xgb_preds, target_names=label_encoder.classes_, zero_division=0)}")

        xgb_results = {"accuracy": xgb_accuracy, "f1_macro": xgb_f1_macro}

    # ── Train full model + feature importance ───────────────────────────────
    model, importance_df = train_full_model(
        X_labeled, y_labeled, available_features, label_encoder
    )

    logger.info(f"\nFeature Importance:")
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")

    # ── Predict on unlabeled windows ────────────────────────────────────────
    if len(unlabeled) > 0:
        X_unlabeled = unlabeled[available_features].values.astype(np.float32)

        if GPU_ML:
            preds = model.predict(cudf.DataFrame(X_unlabeled))
            if hasattr(preds, "to_pandas"):
                preds = preds.to_pandas().values
        else:
            preds = model.predict(X_unlabeled)

        pred_labels = label_encoder.inverse_transform(preds.astype(int))
        unlabeled["predicted_category"] = pred_labels

    # ── Compile output ──────────────────────────────────────────────────────
    labeled["predicted_category"] = label_encoder.inverse_transform(rf_preds) if len(rf_preds) == len(labeled) else labeled["category"]
    labeled["is_labeled"] = True

    if len(unlabeled) > 0:
        unlabeled["is_labeled"] = False
        output_df = pd.concat([labeled, unlabeled], ignore_index=True)
    else:
        output_df = labeled

    # Add evaluation metrics as metadata columns for labeled rows
    output_df["rf_accuracy"] = rf_accuracy
    output_df["rf_f1_macro"] = rf_f1_macro

    # ── Write outputs ───────────────────────────────────────────────────────
    logger.info(f"Writing classifications.parquet ({len(output_df)} rows)...")
    output_df.to_parquet(output_path, index=False, storage_options=storage_options)

    logger.info(f"Writing feature_importance.parquet ({len(importance_df)} rows)...")
    importance_df.to_parquet(importance_path, index=False, storage_options=storage_options)

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("Stage 9 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time:            {elapsed / 60:.1f} minutes")
    logger.info(f"  Labeled events:        {len(labeled)}")
    logger.info(f"  Unlabeled windows:     {len(unlabeled)}")
    logger.info(f"  RF LOOCV accuracy:     {rf_accuracy:.3f}")
    logger.info(f"  RF LOOCV F1 (macro):   {rf_f1_macro:.3f}")
    if xgb_results:
        logger.info(f"  XGB LOOCV accuracy:    {xgb_results['accuracy']:.3f}")
        logger.info(f"  XGB LOOCV F1 (macro):  {xgb_results['f1_macro']:.3f}")
    logger.info(f"  Top feature:           {importance_df.iloc[0]['feature']} "
                f"({importance_df.iloc[0]['importance']:.4f})")
    logger.info(f"  Outputs at:            {S3_INTERMEDIATE}/")


if __name__ == "__main__":
    main()
