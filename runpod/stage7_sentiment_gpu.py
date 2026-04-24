#!/usr/bin/env python3
"""
Stage 7: GPU-Accelerated Sentiment Analysis (Q6)
==================================================
Analyzes sentiment in anomaly-window text vs baseline text using
cardiffnlp/twitter-roberta-base-sentiment-latest on GPU.

Outputs:
  - sentiment.parquet: per-anomaly-window sentiment metrics + shift analysis

Usage: python stage7_sentiment_gpu.py
"""

import os
import sys
import time
import logging
import gc
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import s3fs
from tqdm import tqdm
from scipy import stats as scipy_stats

# ── GPU imports with fallback ───────────────────────────────────────────────
try:
    import cudf
    GPU_AVAILABLE = True
    print("cuDF available - using GPU for data loading")
except ImportError:
    GPU_AVAILABLE = False
    print("cuDF not available - using pandas")

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_DIR = os.environ.get(
    "LOG_DIR",
    "/content/logs" if os.path.isdir("/content") else "/workspace/logs",
)
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "stage7_sentiment.log")),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import S3_BUCKET, S3_BASE, S3_INTERMEDIATE

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
INFERENCE_BATCH_SIZE = int(os.environ.get("SENTIMENT_BATCH_SIZE", "128"))
MAX_TEXT_LENGTH = 512  # RoBERTa max tokens
BASELINE_SAMPLE_SIZE = 5000
ANOMALY_SAMPLE_SIZE = 10000


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


def load_sentiment_pipeline():
    """Load the sentiment analysis pipeline on GPU."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Loading {MODEL_NAME} on {'GPU' if device == 0 else 'CPU'}...")

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=device,
        max_length=MAX_TEXT_LENGTH,
        truncation=True,
        top_k=None,  # return all class probabilities
    )
    logger.info("Sentiment model loaded")
    return sentiment_pipe


def read_anomaly_windows(storage_options):
    """Read anomaly windows from S3."""
    path = f"{S3_INTERMEDIATE}/anomaly_windows.parquet"
    logger.info(f"Reading anomaly windows from {path}")
    df = pd.read_parquet(path, storage_options=storage_options)
    if "window_start" in df.columns:
        df = df.rename(columns={"window_start": "start_time", "window_end": "end_time"})
    logger.info(f"  Loaded {len(df)} anomaly windows")
    return df


def fetch_texts_for_window(subreddit, start_ts, end_ts, storage_options,
                           max_texts=ANOMALY_SAMPLE_SIZE):
    """Fetch raw text from S3 for a specific subreddit and time range."""
    start_dt = pd.Timestamp(start_ts) if isinstance(start_ts, str) else pd.Timestamp(start_ts, unit="s") if isinstance(start_ts, (int, float)) else pd.Timestamp(start_ts)
    end_dt = pd.Timestamp(end_ts) if isinstance(end_ts, str) else pd.Timestamp(end_ts, unit="s") if isinstance(end_ts, (int, float)) else pd.Timestamp(end_ts)

    months_needed = set()
    current = start_dt.replace(day=1)
    while current <= end_dt:
        months_needed.add((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    texts = []
    for year, month in months_needed:
        for data_type, text_col in [("comments", "body"), ("submissions", "selftext")]:
            try:
                path = f"{S3_BASE}/{data_type}/yyyy={year}/mm={month:02d}/"
                df = pd.read_parquet(
                    path,
                    columns=["subreddit", "created_utc", text_col],
                    storage_options=storage_options,
                    engine="pyarrow",
                    filters=[("subreddit", "==", subreddit)],
                )
                if len(df) == 0:
                    continue

                df["ts"] = pd.to_datetime(df["created_utc"], unit="s")
                mask = (df["ts"] >= start_dt) & (df["ts"] <= end_dt)
                filtered = df[mask][text_col].dropna().astype(str)
                filtered = filtered[~filtered.isin(["[deleted]", "[removed]", ""])]
                texts.extend(filtered.tolist())
            except Exception as e:
                logger.debug(f"Could not read {data_type} {year}-{month:02d}: {e}")

        # Also submission titles
        try:
            path = f"{S3_BASE}/submissions/yyyy={year}/mm={month:02d}/"
            df = pd.read_parquet(
                path,
                columns=["subreddit", "created_utc", "title"],
                storage_options=storage_options,
                engine="pyarrow",
                filters=[("subreddit", "==", subreddit)],
            )
            if len(df) > 0:
                df["ts"] = pd.to_datetime(df["created_utc"], unit="s")
                mask = (df["ts"] >= start_dt) & (df["ts"] <= end_dt)
                titles = df[mask]["title"].dropna().astype(str).tolist()
                texts.extend(titles)
        except Exception:
            pass

    # Sample if too many
    if len(texts) > max_texts:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(texts), max_texts, replace=False)
        texts = [texts[i] for i in indices]

    return texts


def fetch_baseline_texts(subreddit, anomaly_start, storage_options,
                         max_texts=BASELINE_SAMPLE_SIZE):
    """Fetch baseline text from a non-anomaly period for the same subreddit.

    Uses a 7-day window starting 14 days before the anomaly.
    """
    start_dt = pd.Timestamp(anomaly_start) if isinstance(anomaly_start, str) else pd.Timestamp(anomaly_start, unit="s") if isinstance(anomaly_start, (int, float)) else pd.Timestamp(anomaly_start)

    baseline_end = start_dt - timedelta(days=7)
    baseline_start = start_dt - timedelta(days=14)

    return fetch_texts_for_window(
        subreddit, baseline_start, baseline_end, storage_options,
        max_texts=max_texts,
    )


def run_sentiment_batch(sentiment_pipe, texts, batch_size=INFERENCE_BATCH_SIZE):
    """Run sentiment inference on a list of texts.

    Returns:
      scores: np.array of sentiment scores (-1 to 1)
      labels: list of label strings
      distributions: list of {neg, neu, pos} probability dicts
    """
    if not texts:
        return np.array([]), [], []

    # Truncate texts to model max
    truncated = [t[:MAX_TEXT_LENGTH * 4] for t in texts]  # rough char limit

    scores = []
    labels = []
    distributions = []

    # Process in batches
    for i in range(0, len(truncated), batch_size):
        batch = truncated[i:i + batch_size]
        try:
            results = sentiment_pipe(batch, batch_size=batch_size)

            for result in results:
                # result is a list of dicts: [{"label": "negative", "score": 0.8}, ...]
                dist = {}
                for item in result:
                    lbl = item["label"].lower()
                    if "neg" in lbl:
                        dist["negative"] = item["score"]
                    elif "neu" in lbl:
                        dist["neutral"] = item["score"]
                    elif "pos" in lbl:
                        dist["positive"] = item["score"]

                # Compute scalar sentiment score: positive - negative
                pos = dist.get("positive", 0)
                neg = dist.get("negative", 0)
                neu = dist.get("neutral", 0)

                score = pos - neg
                label = max(dist, key=dist.get) if dist else "neutral"

                scores.append(score)
                labels.append(label)
                distributions.append(dist)

        except Exception as e:
            logger.warning(f"Batch sentiment error: {e}")
            for _ in batch:
                scores.append(0.0)
                labels.append("neutral")
                distributions.append({"negative": 0.33, "neutral": 0.34, "positive": 0.33})

    return np.array(scores), labels, distributions


def compute_sentiment_stats(scores, distributions):
    """Compute summary statistics from sentiment scores."""
    if len(scores) == 0:
        return {
            "mean_sentiment": np.nan,
            "median_sentiment": np.nan,
            "std_sentiment": np.nan,
            "prop_positive": np.nan,
            "prop_negative": np.nan,
            "prop_neutral": np.nan,
            "n_texts": 0,
        }

    pos_count = sum(1 for s in scores if s > 0.1)
    neg_count = sum(1 for s in scores if s < -0.1)
    neu_count = len(scores) - pos_count - neg_count

    return {
        "mean_sentiment": float(np.mean(scores)),
        "median_sentiment": float(np.median(scores)),
        "std_sentiment": float(np.std(scores)),
        "prop_positive": pos_count / len(scores),
        "prop_negative": neg_count / len(scores),
        "prop_neutral": neu_count / len(scores),
        "n_texts": len(scores),
    }


def compute_sentiment_shift(anomaly_scores, baseline_scores):
    """Compute sentiment shift with statistical significance (Welch t-test)."""
    if len(anomaly_scores) < 2 or len(baseline_scores) < 2:
        return {
            "sentiment_shift": np.nan,
            "shift_tstat": np.nan,
            "shift_pvalue": np.nan,
            "shift_significant": False,
        }

    shift = float(np.mean(anomaly_scores) - np.mean(baseline_scores))

    # Welch's t-test (unequal variance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tstat, pvalue = scipy_stats.ttest_ind(
            anomaly_scores, baseline_scores, equal_var=False
        )

    return {
        "sentiment_shift": shift,
        "shift_tstat": float(tstat) if not np.isnan(tstat) else np.nan,
        "shift_pvalue": float(pvalue) if not np.isnan(pvalue) else np.nan,
        "shift_significant": bool(pvalue < 0.05) if not np.isnan(pvalue) else False,
    }


def main():
    logger.info("=" * 70)
    logger.info("Stage 7: GPU-Accelerated Sentiment Analysis")
    logger.info("=" * 70)
    t_start = time.time()

    storage_options = get_s3_storage_options()
    s3 = get_s3fs_client()

    # Check for existing output
    output_path = f"{S3_INTERMEDIATE}/sentiment.parquet"
    output_exists = False
    try:
        if s3.exists(output_path.replace("s3://", "")):
            logger.info("sentiment.parquet already exists on S3. Skipping.")
            logger.info("Delete the file to force re-run.")
            return
    except Exception:
        pass

    # Load data and model
    anomaly_windows = read_anomaly_windows(storage_options)
    sentiment_pipe = load_sentiment_pipeline()

    # ── Process each anomaly window ─────────────────────────────────────────
    results = []
    n_windows = len(anomaly_windows)

    for idx, (_, row) in enumerate(tqdm(anomaly_windows.iterrows(),
                                         total=n_windows,
                                         desc="Sentiment analysis")):
        window_id = row.get("window_id", idx)
        subreddit = row["subreddit"]
        start_ts = row["start_time"]
        end_ts = row["end_time"]

        logger.info(f"Window {window_id}: r/{subreddit} "
                    f"({start_ts} to {end_ts})")

        try:
            # ── Anomaly-window sentiment ────────────────────────────────────
            anomaly_texts = fetch_texts_for_window(
                subreddit, start_ts, end_ts, storage_options,
                max_texts=ANOMALY_SAMPLE_SIZE,
            )
            logger.info(f"  Anomaly texts: {len(anomaly_texts)}")

            if len(anomaly_texts) == 0:
                logger.warning(f"  No anomaly texts for window {window_id}")
                continue

            anomaly_scores, anomaly_labels, anomaly_dists = run_sentiment_batch(
                sentiment_pipe, anomaly_texts
            )
            anomaly_stats = compute_sentiment_stats(anomaly_scores, anomaly_dists)

            # ── Baseline sentiment ──────────────────────────────────────────
            baseline_texts = fetch_baseline_texts(
                subreddit, start_ts, storage_options,
                max_texts=BASELINE_SAMPLE_SIZE,
            )
            logger.info(f"  Baseline texts: {len(baseline_texts)}")

            baseline_scores = np.array([])
            baseline_stats = compute_sentiment_stats(np.array([]), [])
            shift_stats = compute_sentiment_shift(anomaly_scores, np.array([]))

            if len(baseline_texts) > 0:
                baseline_scores, _, baseline_dists = run_sentiment_batch(
                    sentiment_pipe, baseline_texts
                )
                baseline_stats = compute_sentiment_stats(baseline_scores, baseline_dists)
                shift_stats = compute_sentiment_shift(anomaly_scores, baseline_scores)

            # ── Compile result ──────────────────────────────────────────────
            result_row = {
                "window_id": window_id,
                "subreddit": subreddit,
                "start_time": start_ts,
                "end_time": end_ts,
                # Anomaly sentiment
                "anomaly_mean_sentiment": anomaly_stats["mean_sentiment"],
                "anomaly_median_sentiment": anomaly_stats["median_sentiment"],
                "anomaly_std_sentiment": anomaly_stats["std_sentiment"],
                "anomaly_prop_positive": anomaly_stats["prop_positive"],
                "anomaly_prop_negative": anomaly_stats["prop_negative"],
                "anomaly_prop_neutral": anomaly_stats["prop_neutral"],
                "anomaly_n_texts": anomaly_stats["n_texts"],
                # Baseline sentiment
                "baseline_mean_sentiment": baseline_stats["mean_sentiment"],
                "baseline_std_sentiment": baseline_stats["std_sentiment"],
                "baseline_prop_positive": baseline_stats["prop_positive"],
                "baseline_prop_negative": baseline_stats["prop_negative"],
                "baseline_n_texts": baseline_stats["n_texts"],
                # Shift
                "sentiment_shift": shift_stats["sentiment_shift"],
                "shift_tstat": shift_stats["shift_tstat"],
                "shift_pvalue": shift_stats["shift_pvalue"],
                "shift_significant": shift_stats["shift_significant"],
            }
            results.append(result_row)

        except Exception as e:
            logger.error(f"Error processing window {window_id}: {e}")
            continue

        # Periodic checkpoint
        if (idx + 1) % 25 == 0:
            logger.info(f"  Progress: {idx + 1}/{n_windows} windows complete")
            gc.collect()

    # ── Build output DataFrame ──────────────────────────────────────────────
    if not results:
        logger.error("No sentiment results. Check anomaly windows and S3 data.")
        sys.exit(1)

    sentiment_df = pd.DataFrame(results)

    # ── Write output to S3 ──────────────────────────────────────────────────
    logger.info(f"Writing sentiment.parquet ({len(sentiment_df)} rows)...")
    sentiment_df.to_parquet(output_path, index=False, storage_options=storage_options)

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("Stage 7 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time:              {elapsed / 60:.1f} minutes")
    logger.info(f"  Windows analyzed:        {len(sentiment_df)}")
    logger.info(f"  Mean anomaly sentiment:  {sentiment_df['anomaly_mean_sentiment'].mean():.4f}")
    logger.info(f"  Mean baseline sentiment: {sentiment_df['baseline_mean_sentiment'].mean():.4f}")
    logger.info(f"  Mean sentiment shift:    {sentiment_df['sentiment_shift'].mean():.4f}")

    n_sig = sentiment_df["shift_significant"].sum()
    logger.info(f"  Significant shifts:      {n_sig}/{len(sentiment_df)} "
                f"({100 * n_sig / len(sentiment_df):.1f}%)")

    # Breakdown by shift direction
    pos_shift = (sentiment_df["sentiment_shift"] > 0).sum()
    neg_shift = (sentiment_df["sentiment_shift"] < 0).sum()
    logger.info(f"  Positive shifts:         {pos_shift}")
    logger.info(f"  Negative shifts:         {neg_shift}")
    logger.info(f"  Output at:               {output_path}")


if __name__ == "__main__":
    main()
