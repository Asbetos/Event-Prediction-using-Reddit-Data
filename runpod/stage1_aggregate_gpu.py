#!/usr/bin/env python3
"""
Stage 1: GPU-Accelerated Aggregation of 478 GB Reddit Data
============================================================
Reads raw parquet from S3 month-by-month using cuDF on A100 80GB.
Produces small intermediate tables:
  - hourly_counts.parquet   (subreddit x hour: post_count, unique_authors, mean_score)
  - daily_counts.parquet    (subreddit x day: same metrics)
  - subreddit_stats.parquet (per-subreddit summary stats)

Outputs written to s3://ven-bda-s3-v2/reddit-data/intermediate/

Usage: python stage1_aggregate_gpu.py
"""

import os
import sys
import time
import logging
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import s3fs
from tqdm import tqdm

# ── GPU imports with fallback ───────────────────────────────────────────────
# Force pandas due to CUDA driver version mismatch on Lightning.ai
GPU_AVAILABLE = False
DF_LIB = pd
print("Using pandas (GPU disabled due to driver version mismatch)")

# ── Logging setup ───────────────────────────────────────────────────────────
import os
LOG_DIR = os.path.expanduser("~/logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
handlers=[
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(os.path.join(LOG_DIR, "stage1_aggregate.log")),
],
)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import S3_BUCKET, S3_BASE, S3_INTERMEDIATE, MONTHS, TOP_N_SUBREDDITS


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


def check_s3_output_exists(s3, path):
    """Check if an output file already exists on S3."""
    try:
        s3_path = path.replace("s3://", "")
        return s3.exists(s3_path)
    except Exception:
        return False


def read_month_parquet(data_type, year, month, storage_options):
    """Read one month of parquet data from S3 into a cuDF/pandas DataFrame.

    Reads only the columns we need for aggregation to minimize memory.
    """
    path = f"{S3_BASE}/{data_type}/yyyy={year}/mm={month:02d}/"

    # Columns we need
    if data_type == "comments":
        columns = ["subreddit", "created_utc", "author", "score"]
    else:  # submissions
        columns = ["subreddit", "created_utc", "author", "score", "num_comments"]

    try:
        if GPU_AVAILABLE:
            df = cudf.read_parquet(
                path,
                columns=columns,
                storage_options=storage_options,
            )
        else:
            df = pd.read_parquet(
                path,
                columns=columns,
                storage_options=storage_options,
                engine="pyarrow",
            )
        return df
    except Exception as e:
        logger.error(f"Failed to read {data_type} for {year}-{month:02d}: {e}")
        return None


def aggregate_hourly(df, data_type):
    """Compute hourly aggregates per subreddit from a month of data.

    Returns a small DataFrame with columns:
      subreddit, hour_bucket, post_count, unique_authors, mean_score, data_type
    """
    if df is None or len(df) == 0:
        return None

    # Convert epoch seconds to datetime and truncate to hour
    if GPU_AVAILABLE:
        df["datetime"] = cudf.to_datetime(df["created_utc"], unit="s")
        df["hour_bucket"] = df["datetime"].dt.floor("h")
    else:
        df["datetime"] = pd.to_datetime(df["created_utc"], unit="s")
        df["hour_bucket"] = df["datetime"].dt.floor("h")

    # Drop datetime column to save memory
    df.drop(columns=["datetime", "created_utc"], inplace=True)

    # GroupBy subreddit + hour
    grouped = df.groupby(["subreddit", "hour_bucket"]).agg(
        post_count=("score", "count"),
        unique_authors=("author", "nunique"),
        mean_score=("score", "mean"),
    ).reset_index()

    grouped["data_type"] = data_type
    return grouped


def aggregate_daily(hourly_df):
    """Roll up hourly counts to daily counts."""
    if hourly_df is None or len(hourly_df) == 0:
        return None

    if GPU_AVAILABLE:
        hourly_df["date"] = hourly_df["hour_bucket"].dt.floor("D")
    else:
        hourly_df["date"] = hourly_df["hour_bucket"].dt.floor("D")

    daily = hourly_df.groupby(["subreddit", "date", "data_type"]).agg(
        post_count=("post_count", "sum"),
        unique_authors=("unique_authors", "sum"),  # approximate (overcounts across hours)
        mean_score=("mean_score", "mean"),
    ).reset_index()

    return daily


def process_one_month(year, month, storage_options):
    """Process one month: read comments + submissions, compute hourly aggs."""
    logger.info(f"Processing {year}-{month:02d}...")
    t0 = time.time()

    hourly_parts = []

    for data_type in ["comments", "submissions"]:
        logger.info(f"  Reading {data_type} for {year}-{month:02d}...")
        df = read_month_parquet(data_type, year, month, storage_options)

        if df is not None:
            n_rows = len(df)
            mem_mb = df.memory_usage(deep=False).sum() / (1024 ** 2)
            logger.info(f"    Loaded {n_rows:,} rows ({mem_mb:.0f} MB)")

            logger.info(f"    Aggregating {data_type} hourly...")
            hourly = aggregate_hourly(df, data_type)

            # Free raw data immediately
            del df
            gc.collect()
            if GPU_AVAILABLE:
                # Safe GPU cleanup — NEVER use rmm.reinitialize() as it
                # resets the entire GPU memory manager and will crash
                # other CUDA processes on this pod (eagle3, sweep, etc.)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            if hourly is not None:
                logger.info(f"    Aggregated to {len(hourly):,} rows")
                hourly_parts.append(hourly)

    if not hourly_parts:
        logger.warning(f"  No data for {year}-{month:02d}")
        return None

    # Concatenate comments + submissions hourly aggs for this month
    if GPU_AVAILABLE:
        month_hourly = cudf.concat(hourly_parts, ignore_index=True)
    else:
        month_hourly = pd.concat(hourly_parts, ignore_index=True)

    elapsed = time.time() - t0
    logger.info(f"  {year}-{month:02d} done in {elapsed:.1f}s "
                f"({len(month_hourly):,} hourly rows)")

    return month_hourly


def compute_subreddit_stats(hourly_all):
    """Compute per-subreddit summary statistics."""
    stats = hourly_all.groupby(["subreddit", "data_type"]).agg(
        total_posts=("post_count", "sum"),
        total_hours_active=("post_count", "count"),
        mean_hourly_posts=("post_count", "mean"),
        max_hourly_posts=("post_count", "max"),
        std_hourly_posts=("post_count", "std"),
        mean_unique_authors=("unique_authors", "mean"),
        overall_mean_score=("mean_score", "mean"),
    ).reset_index()

    return stats


def filter_top_subreddits(hourly_all, top_n=TOP_N_SUBREDDITS):
    """Filter to top N subreddits by total post volume (comments + submissions)."""
    # Sum across data types
    volume = hourly_all.groupby("subreddit").agg(
        total=("post_count", "sum")
    ).reset_index()

    if GPU_AVAILABLE:
        volume = volume.sort_values("total", ascending=False)
        top_subs = volume.head(top_n)["subreddit"]
        # cuDF merge-based filter
        filtered = hourly_all.merge(
            top_subs.to_frame(), on="subreddit", how="inner"
        )
    else:
        volume = volume.sort_values("total", ascending=False)
        top_subs = volume.head(top_n)["subreddit"].tolist()
        filtered = hourly_all[hourly_all["subreddit"].isin(top_subs)]

    logger.info(f"Filtered from {hourly_all['subreddit'].nunique()} to "
                f"{filtered['subreddit'].nunique()} subreddits (top {top_n})")
    return filtered


def to_pandas(df):
    """Convert cuDF DataFrame to pandas for writing."""
    if GPU_AVAILABLE and hasattr(df, "to_pandas"):
        return df.to_pandas()
    return df


def write_parquet_to_s3(df, s3_path, storage_options):
    """Write a DataFrame to S3 as parquet."""
    pdf = to_pandas(df)
    logger.info(f"Writing {len(pdf):,} rows to {s3_path}")
    pdf.to_parquet(s3_path, index=False, storage_options=storage_options)
    logger.info(f"  Written successfully ({pdf.memory_usage(deep=True).sum() / 1024**2:.1f} MB)")


def main():
    logger.info("=" * 70)
    logger.info("Stage 1: GPU-Accelerated Reddit Data Aggregation")
    logger.info("=" * 70)
    t_start = time.time()

    storage_options = get_s3_storage_options()
    s3 = get_s3fs_client()

    # Check for existing outputs (checkpointing)
    hourly_path = f"{S3_INTERMEDIATE}/hourly_counts.parquet"
    daily_path = f"{S3_INTERMEDIATE}/daily_counts.parquet"
    stats_path = f"{S3_INTERMEDIATE}/subreddit_stats.parquet"

    if (check_s3_output_exists(s3, hourly_path) and
        check_s3_output_exists(s3, daily_path) and
        check_s3_output_exists(s3, stats_path)):
        logger.info("All Stage 1 outputs already exist on S3. Skipping.")
        logger.info("Delete the intermediate files to force re-run.")
        return

    # ── Process each month ──────────────────────────────────────────────────
    all_hourly = []
    checkpoint_path = f"{S3_INTERMEDIATE}/stage1_checkpoint/"

    for year, month in tqdm(MONTHS, desc="Processing months"):
        # Check monthly checkpoint
        month_ckpt = f"{checkpoint_path}{year}_{month:02d}.parquet"
        if check_s3_output_exists(s3, month_ckpt):
            logger.info(f"Loading checkpoint for {year}-{month:02d}")
            if GPU_AVAILABLE:
                month_hourly = cudf.read_parquet(
                    month_ckpt, storage_options=storage_options
                )
            else:
                month_hourly = pd.read_parquet(
                    month_ckpt, storage_options=storage_options
                )
            all_hourly.append(month_hourly)
            continue

        month_hourly = process_one_month(year, month, storage_options)

        if month_hourly is not None:
            # Save monthly checkpoint
            try:
                write_parquet_to_s3(month_hourly, month_ckpt, storage_options)
                logger.info(f"  Checkpoint saved: {month_ckpt}")
            except Exception as e:
                logger.warning(f"  Could not save checkpoint: {e}")

            all_hourly.append(month_hourly)

        # Force garbage collection between months
        gc.collect()

    if not all_hourly:
        logger.error("No data was processed. Check S3 access and paths.")
        sys.exit(1)

    # ── Concatenate all months ──────────────────────────────────────────────
    logger.info("Concatenating all months...")
    if GPU_AVAILABLE:
        hourly_all = cudf.concat(all_hourly, ignore_index=True)
    else:
        hourly_all = pd.concat(all_hourly, ignore_index=True)

    del all_hourly
    gc.collect()

    logger.info(f"Total hourly rows: {len(hourly_all):,}")
    logger.info(f"Total subreddits: {hourly_all['subreddit'].nunique()}")

    # ── Filter to top subreddits ────────────────────────────────────────────
    logger.info(f"Filtering to top {TOP_N_SUBREDDITS} subreddits by volume...")
    hourly_filtered = filter_top_subreddits(hourly_all, TOP_N_SUBREDDITS)
    del hourly_all
    gc.collect()

    # ── Compute daily counts ────────────────────────────────────────────────
    logger.info("Computing daily counts...")
    daily_counts = aggregate_daily(hourly_filtered)

    # ── Compute subreddit stats ─────────────────────────────────────────────
    logger.info("Computing subreddit stats...")
    subreddit_stats = compute_subreddit_stats(hourly_filtered)

    # ── Write outputs to S3 ─────────────────────────────────────────────────
    logger.info("Writing outputs to S3...")
    write_parquet_to_s3(hourly_filtered, hourly_path, storage_options)
    write_parquet_to_s3(daily_counts, daily_path, storage_options)
    write_parquet_to_s3(subreddit_stats, stats_path, storage_options)

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("Stage 1 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time:        {elapsed / 60:.1f} minutes")
    logger.info(f"  Hourly rows:       {len(hourly_filtered):,}")
    logger.info(f"  Daily rows:        {len(daily_counts):,}")
    logger.info(f"  Subreddits:        {hourly_filtered['subreddit'].nunique()}")
    logger.info(f"  Date range:        "
                f"{to_pandas(hourly_filtered)['hour_bucket'].min()} to "
                f"{to_pandas(hourly_filtered)['hour_bucket'].max()}")
    logger.info(f"  Outputs at:        {S3_INTERMEDIATE}/")
    logger.info(f"    - hourly_counts.parquet")
    logger.info(f"    - daily_counts.parquet")
    logger.info(f"    - subreddit_stats.parquet")


if __name__ == "__main__":
    main()
