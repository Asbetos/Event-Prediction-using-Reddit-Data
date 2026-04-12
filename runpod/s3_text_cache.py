#!/usr/bin/env python3
"""
Shared S3 text cache for RunPod GPU stages (6, 7, 8).

Downloads each month's parquet text data ONCE to /workspace/s3_cache/ and
serves all NLP stages from local disk — eliminating redundant S3 reads.

Without this cache, stages 6/7/8 each download the same month data separately,
tripling S3 I/O. Stage 7 is worst: it reads per-window (hundreds of S3 calls).

Standalone usage (pre-fetch all months with 16 parallel workers):
    python s3_text_cache.py

Library usage:
    from s3_text_cache import S3TextCache
    cache = S3TextCache(storage_options)
    cache.prefetch(anomaly_windows)
    texts = cache.get_texts_for_range(subreddit, start_ts, end_ts, max_texts=10000)
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)

S3_BUCKET = "ven-bda-s3-v2"
S3_BASE = f"s3://{S3_BUCKET}/reddit-data/parquet"
S3_INTERMEDIATE = f"s3://{S3_BUCKET}/reddit-data/intermediate"
CACHE_DIR = "/workspace/s3_cache"
MONTHS = [(2023, m) for m in range(6, 13)] + [(2024, m) for m in range(1, 8)]


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


class S3TextCache:
    """Thread-safe S3 text data cache backed by local parquet on /workspace.

    Each month's text data (comments body, submission selftext, submission title)
    is downloaded once, filtered to anomaly-window subreddits, and stored as a
    single parquet file. Subsequent reads are pure local disk I/O.
    """

    def __init__(self, storage_options, cache_dir=CACHE_DIR):
        self.storage_options = storage_options
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._loaded = {}  # in-memory LRU: (year, month) -> DataFrame

    def _cache_path(self, year, month):
        return os.path.join(self.cache_dir, f"text_{year}_{month:02d}.parquet")

    def _download_month(self, year, month, subreddits):
        """Download one month of text from S3, filtered to given subreddits."""
        cache_path = self._cache_path(year, month)
        if os.path.exists(cache_path):
            return True

        sub_filter = [("subreddit", "in", subreddits)]
        all_dfs = []

        for data_type, text_col in [("comments", "body"), ("submissions", "selftext")]:
            try:
                path = f"{S3_BASE}/{data_type}/yyyy={year}/mm={month:02d}/"
                cols = ["subreddit", "created_utc", text_col]
                df = pd.read_parquet(
                    path, columns=cols,
                    storage_options=self.storage_options,
                    engine="pyarrow",
                    filters=sub_filter,
                )
                if len(df) > 0:
                    df["ts"] = pd.to_datetime(df["created_utc"], unit="s")
                    df["text"] = df[text_col].fillna("")
                    mask = ~df["text"].isin(["[deleted]", "[removed]", ""])
                    all_dfs.append(df[mask][["subreddit", "ts", "text"]].copy())
                    del df
            except Exception as e:
                logger.warning(f"  Could not read {data_type} {year}-{month:02d}: {e}")

        # Submission titles (separate read since different column)
        try:
            path = f"{S3_BASE}/submissions/yyyy={year}/mm={month:02d}/"
            df = pd.read_parquet(
                path, columns=["subreddit", "created_utc", "title"],
                storage_options=self.storage_options,
                engine="pyarrow",
                filters=sub_filter,
            )
            if len(df) > 0:
                df["ts"] = pd.to_datetime(df["created_utc"], unit="s")
                df["text"] = df["title"].fillna("")
                mask = df["text"] != ""
                all_dfs.append(df[mask][["subreddit", "ts", "text"]].copy())
                del df
        except Exception:
            pass

        if all_dfs:
            result = pd.concat(all_dfs, ignore_index=True)
            result.to_parquet(cache_path, index=False)
            size_mb = os.path.getsize(cache_path) / 1024**2
            logger.info(f"  Cached {year}-{month:02d}: {len(result):,} rows ({size_mb:.0f} MB)")
        else:
            pd.DataFrame(columns=["subreddit", "ts", "text"]).to_parquet(
                cache_path, index=False)
            logger.info(f"  Cached {year}-{month:02d}: empty")

        return True

    def prefetch(self, anomaly_windows, n_workers=16):
        """Pre-fetch all months in parallel using ThreadPoolExecutor.

        Uses up to 16 threads for concurrent S3 downloads — fully utilizes
        network bandwidth on the 128-core EPYC without touching the GPU.
        """
        subreddits = anomaly_windows["subreddit"].unique().tolist()
        logger.info(f"Pre-fetching {len(MONTHS)} months for {len(subreddits)} subreddits "
                    f"using {n_workers} threads...")

        to_download = [(y, m) for y, m in MONTHS
                       if not os.path.exists(self._cache_path(y, m))]

        if not to_download:
            logger.info("  All months already cached on /workspace!")
            return

        logger.info(f"  Downloading {len(to_download)}/{len(MONTHS)} months...")
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self._download_month, y, m, subreddits): (y, m)
                for y, m in to_download
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                              desc="Caching S3 text data"):
                year, month = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"  FAILED {year}-{month:02d}: {e}")

        elapsed = time.time() - t0
        logger.info(f"  Pre-fetch complete in {elapsed:.0f}s")

    def get_month(self, year, month):
        """Load cached month data into memory. Returns DataFrame [subreddit, ts, text]."""
        key = (year, month)
        if key in self._loaded:
            return self._loaded[key]

        cache_path = self._cache_path(year, month)
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            if "ts" in df.columns and len(df) > 0:
                df["ts"] = pd.to_datetime(df["ts"])
            self._loaded[key] = df
            return df

        return pd.DataFrame(columns=["subreddit", "ts", "text"])

    def unload_month(self, year, month):
        """Free in-memory data for a month (disk cache remains)."""
        self._loaded.pop((year, month), None)

    def get_window_texts(self, month_df, subreddit, start_ts, end_ts, max_texts=None):
        """Fast local filter: get texts for one anomaly window from a loaded month."""
        if len(month_df) == 0:
            return []
        start_dt = pd.Timestamp(start_ts)
        end_dt = pd.Timestamp(end_ts)
        mask = (
            (month_df["subreddit"] == subreddit) &
            (month_df["ts"] >= start_dt) &
            (month_df["ts"] <= end_dt)
        )
        texts = month_df.loc[mask, "text"].tolist()

        if max_texts and len(texts) > max_texts:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(texts), max_texts, replace=False)
            texts = [texts[i] for i in indices]

        return texts

    def get_texts_for_range(self, subreddit, start_ts, end_ts, max_texts=None):
        """Get texts spanning multiple months. Auto-loads needed months from cache."""
        start_dt = pd.Timestamp(start_ts) if not isinstance(start_ts, pd.Timestamp) else start_ts
        end_dt = pd.Timestamp(end_ts) if not isinstance(end_ts, pd.Timestamp) else end_ts

        # Determine which months we need
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
            month_df = self.get_month(year, month)
            if len(month_df) > 0:
                window_texts = self.get_window_texts(
                    month_df, subreddit, start_dt, end_dt)
                texts.extend(window_texts)

        if max_texts and len(texts) > max_texts:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(texts), max_texts, replace=False)
            texts = [texts[i] for i in indices]

        return texts


if __name__ == "__main__":
    """Pre-fetch all S3 text data for anomaly-window processing."""
    os.makedirs("/workspace/logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("/workspace/logs/s3_cache.log"),
        ],
    )

    storage_options = get_s3_storage_options()

    logger.info("=" * 60)
    logger.info("S3 Text Cache — Pre-fetch All Months")
    logger.info("=" * 60)

    logger.info("Loading anomaly windows from S3...")
    anomaly_windows = pd.read_parquet(
        f"{S3_INTERMEDIATE}/anomaly_windows.parquet",
        storage_options=storage_options,
    )
    if "window_start" in anomaly_windows.columns:
        anomaly_windows = anomaly_windows.rename(
            columns={"window_start": "start_time", "window_end": "end_time"})

    logger.info(f"Found {len(anomaly_windows)} anomaly windows across "
               f"{anomaly_windows['subreddit'].nunique()} subreddits")

    cache = S3TextCache(storage_options)
    cache.prefetch(anomaly_windows, n_workers=16)

    # Report cache size
    total_size = sum(
        os.path.getsize(os.path.join(CACHE_DIR, f))
        for f in os.listdir(CACHE_DIR) if f.endswith(".parquet")
    )
    logger.info(f"Total cache size: {total_size / 1024**3:.1f} GB at {CACHE_DIR}")
    logger.info("Done — stages 6, 7, 8 will now read from local cache.")
