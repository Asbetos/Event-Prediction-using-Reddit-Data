#!/usr/bin/env python3
"""
Stage 6: GPU-Accelerated Named Entity Recognition (Q5)
========================================================
Reads anomaly windows from Stage 2 (EC2), fetches raw text from S3 for each
window, and extracts named entities using spaCy en_core_web_trf (GPU).

Optimized: reads each month's parquet ONCE, then matches all windows in that
month (avoids redundant S3 reads).

Outputs:
  - entities.parquet: entity counts per anomaly window
  - entity_cooccurrence.parquet: entity pair co-occurrences

Usage: python stage6_ner_gpu.py
"""

import os
import sys
import time
import logging
import gc
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import s3fs
from tqdm import tqdm

# ── GPU imports with fallback ───────────────────────────────────────────────
try:
    import cudf
    GPU_AVAILABLE = True
    print("cuDF available - using GPU for data loading")
except ImportError:
    GPU_AVAILABLE = False
    print("cuDF not available - using pandas")

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("stage6_ner.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
S3_BUCKET = "ven-bda-s3-v2"
S3_BASE = f"s3://{S3_BUCKET}/reddit-data/parquet"
S3_INTERMEDIATE = f"s3://{S3_BUCKET}/reddit-data/intermediate"
NER_BATCH_SIZE = 10_000
ENTITY_TYPES = {"PERSON", "ORG", "GPE"}
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


def get_s3fs_client():
    """Create an s3fs filesystem client."""
    opts = get_s3_storage_options()
    if opts:
        return s3fs.S3FileSystem(key=opts.get("key"), secret=opts.get("secret"),
                                  token=opts.get("token"))
    return s3fs.S3FileSystem()


def load_spacy_model():
    """Load spaCy en_core_web_trf with GPU if available."""
    import spacy

    # Try to activate GPU
    try:
        spacy.prefer_gpu()
        logger.info("spaCy GPU activated")
    except Exception:
        logger.info("spaCy running on CPU")

    logger.info("Loading en_core_web_trf model...")
    nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "lemmatizer"])
    # Increase max length for long Reddit posts
    nlp.max_length = 1_500_000
    logger.info("spaCy model loaded")
    return nlp


def read_anomaly_windows(storage_options):
    """Read anomaly windows produced by EC2 Stage 2."""
    path = f"{S3_INTERMEDIATE}/anomaly_windows.parquet"
    logger.info(f"Reading anomaly windows from {path}")
    df = pd.read_parquet(path, storage_options=storage_options)
    # Normalize column names from Stage 2 (window_start/window_end -> start_time/end_time)
    if "window_start" in df.columns:
        df = df.rename(columns={"window_start": "start_time", "window_end": "end_time"})
    logger.info(f"  Loaded {len(df)} anomaly windows")
    return df


def read_month_text(year, month, storage_options, subreddits):
    """Read one month of text data from S3, filtered to only the given subreddits.
    Returns a DataFrame with subreddit, timestamp, and text columns."""
    all_dfs = []
    sub_filter = [("subreddit", "in", subreddits)]

    for data_type in ["comments", "submissions"]:
        path = f"{S3_BASE}/{data_type}/yyyy={year}/mm={month:02d}/"
        text_col = "body" if data_type == "comments" else "selftext"

        try:
            cols = ["subreddit", "created_utc", text_col]
            df = pd.read_parquet(
                path,
                columns=cols,
                storage_options=storage_options,
                engine="pyarrow",
                filters=sub_filter,
            )
            if len(df) == 0:
                continue

            df["ts"] = pd.to_datetime(df["created_utc"], unit="s")
            df["text"] = df[text_col].fillna("")
            # Filter out deleted/removed/empty
            mask = ~df["text"].isin(["[deleted]", "[removed]", ""])
            df = df[mask][["subreddit", "ts", "text"]].copy()
            all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not read {data_type} {year}-{month:02d}: {e}")

    # Also grab submission titles
    try:
        path = f"{S3_BASE}/submissions/yyyy={year}/mm={month:02d}/"
        df = pd.read_parquet(
            path,
            columns=["subreddit", "created_utc", "title"],
            storage_options=storage_options,
            engine="pyarrow",
            filters=sub_filter,
        )
        if len(df) > 0:
            df["ts"] = pd.to_datetime(df["created_utc"], unit="s")
            df["text"] = df["title"].fillna("")
            mask = df["text"] != ""
            all_dfs.append(df[mask][["subreddit", "ts", "text"]].copy())
    except Exception:
        pass

    if not all_dfs:
        return pd.DataFrame(columns=["subreddit", "ts", "text"])

    result = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"  Month {year}-{month:02d}: {len(result):,} text rows loaded")
    return result


def get_texts_for_window(month_df, subreddit, start_ts, end_ts):
    """Filter month DataFrame for a specific window. Fast pandas ops."""
    start_dt = pd.Timestamp(start_ts)
    end_dt = pd.Timestamp(end_ts)
    mask = (
        (month_df["subreddit"] == subreddit) &
        (month_df["ts"] >= start_dt) &
        (month_df["ts"] <= end_dt)
    )
    return month_df.loc[mask, "text"].tolist()


def extract_entities_batch(nlp, texts, batch_size=NER_BATCH_SIZE):
    """Run NER on a batch of texts using spacy.pipe for efficiency."""
    entity_counts = Counter()
    cooccurrence = Counter()

    # Truncate very long texts to avoid memory issues
    truncated = [t[:5000] if len(t) > 5000 else t for t in texts]

    processed = 0
    for doc in nlp.pipe(truncated, batch_size=batch_size):
        doc_entities = set()
        for ent in doc.ents:
            if ent.label_ in ENTITY_TYPES:
                ent_text = ent.text.strip()
                if len(ent_text) > 1:
                    key = (ent_text, ent.label_)
                    entity_counts[key] += 1
                    doc_entities.add(key)

        if len(doc_entities) >= 2:
            for pair in combinations(sorted(doc_entities), 2):
                cooccurrence[pair] += 1

        processed += 1

    return entity_counts, cooccurrence


def main():
    logger.info("=" * 70)
    logger.info("Stage 6: GPU-Accelerated Named Entity Recognition")
    logger.info("=" * 70)
    t_start = time.time()

    storage_options = get_s3_storage_options()
    s3 = get_s3fs_client()

    # Check for existing output
    output_path = f"{S3_INTERMEDIATE}/entities.parquet"
    cooc_path = f"{S3_INTERMEDIATE}/entity_cooccurrence.parquet"
    try:
        if s3.exists(output_path.replace("s3://", "")):
            logger.info("entities.parquet already exists on S3. Skipping.")
            logger.info("Delete the file to force re-run.")
            return
    except Exception:
        pass

    # Load anomaly windows
    anomaly_windows = read_anomaly_windows(storage_options)

    # Parse timestamps
    anomaly_windows["start_dt"] = pd.to_datetime(anomaly_windows["start_time"])
    anomaly_windows["end_dt"] = pd.to_datetime(anomaly_windows["end_time"])
    anomaly_windows["start_month"] = (
        anomaly_windows["start_dt"].dt.year * 100 + anomaly_windows["start_dt"].dt.month
    )

    # Load spaCy model
    nlp = load_spacy_model()

    # ── Process month-by-month ──────────────────────────────────────────────
    all_entity_rows = []
    all_cooc_rows = []
    windows_processed = 0
    windows_with_entities = 0

    for year, month in tqdm(MONTHS, desc="Processing months"):
        month_key = year * 100 + month

        # Get windows that start in this month
        month_windows = anomaly_windows[anomaly_windows["start_month"] == month_key]
        if len(month_windows) == 0:
            logger.info(f"  No anomaly windows in {year}-{month:02d}, skipping")
            continue

        # Get unique subreddits needed for this month
        month_subreddits = month_windows["subreddit"].unique().tolist()
        logger.info(f"Processing {year}-{month:02d}: {len(month_windows)} windows, "
                     f"{len(month_subreddits)} subreddits")

        # Read this month's text data ONCE, filtered to relevant subreddits
        month_df = read_month_text(year, month, storage_options, month_subreddits)
        if len(month_df) == 0:
            logger.warning(f"  No text data for {year}-{month:02d}")
            windows_processed += len(month_windows)
            continue

        # Index by subreddit for faster lookups
        month_df_indexed = month_df.set_index("subreddit", drop=False)

        # Process each window in this month
        for _, row in month_windows.iterrows():
            window_id = row.get("window_id", windows_processed)
            subreddit = row["subreddit"]

            texts = get_texts_for_window(month_df, subreddit, row["start_dt"], row["end_dt"])

            if len(texts) == 0:
                windows_processed += 1
                continue

            # Cap at 100k texts
            if len(texts) > 100_000:
                rng = np.random.RandomState(42)
                indices = rng.choice(len(texts), 100_000, replace=False)
                texts = [texts[i] for i in indices]

            # Run NER
            try:
                entity_counts, cooccurrence = extract_entities_batch(nlp, texts)
            except Exception as e:
                logger.error(f"NER error on window {window_id}: {e}")
                windows_processed += 1
                continue

            if entity_counts:
                windows_with_entities += 1
                for (ent_text, ent_label), count in entity_counts.most_common(200):
                    all_entity_rows.append({
                        "window_id": window_id,
                        "subreddit": subreddit,
                        "entity_text": ent_text,
                        "entity_label": ent_label,
                        "count": count,
                    })

                for (ent1, ent2), count in cooccurrence.most_common(100):
                    all_cooc_rows.append({
                        "window_id": window_id,
                        "entity1_text": ent1[0],
                        "entity1_label": ent1[1],
                        "entity2_text": ent2[0],
                        "entity2_label": ent2[1],
                        "cooccurrence_count": count,
                    })

            windows_processed += 1

        # Free month data
        del month_df, month_df_indexed
        gc.collect()

        logger.info(f"  Month done. Total windows: {windows_processed}, "
                     f"with entities: {windows_with_entities}, "
                     f"entity rows: {len(all_entity_rows)}")

    # ── Build output DataFrames ─────────────────────────────────────────────
    if not all_entity_rows:
        logger.error("No entities extracted. Check anomaly windows and S3 data.")
        sys.exit(1)

    entities_df = pd.DataFrame(all_entity_rows)
    cooc_df = pd.DataFrame(all_cooc_rows) if all_cooc_rows else pd.DataFrame()

    # ── Write outputs to S3 ─────────────────────────────────────────────────
    logger.info(f"Writing entities.parquet ({len(entities_df)} rows)...")
    entities_df.to_parquet(output_path, index=False, storage_options=storage_options)

    if len(cooc_df) > 0:
        logger.info(f"Writing entity_cooccurrence.parquet ({len(cooc_df)} rows)...")
        cooc_df.to_parquet(cooc_path, index=False, storage_options=storage_options)

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("Stage 6 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time:           {elapsed / 60:.1f} minutes")
    logger.info(f"  Windows processed:    {windows_processed}")
    logger.info(f"  Windows with entities:{windows_with_entities}")
    logger.info(f"  Total entity rows:    {len(entities_df):,}")
    logger.info(f"  Unique entities:      {entities_df['entity_text'].nunique():,}")

    # Top entities by type
    for label in ENTITY_TYPES:
        subset = entities_df[entities_df["entity_label"] == label]
        if len(subset) > 0:
            top = (subset.groupby("entity_text")["count"]
                   .sum().sort_values(ascending=False).head(10))
            logger.info(f"  Top {label} entities:")
            for ent, cnt in top.items():
                logger.info(f"    {ent}: {cnt}")

    logger.info(f"  Co-occurrence pairs:  {len(cooc_df):,}")
    logger.info(f"  Outputs at:           {S3_INTERMEDIATE}/")


if __name__ == "__main__":
    main()
