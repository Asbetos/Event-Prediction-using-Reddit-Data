#!/usr/bin/env python3
"""
Stage 6: GPU-Accelerated Named Entity Recognition (Q5)
========================================================
Reads anomaly windows from Stage 2 (EC2), fetches raw text from S3 for each
window, and extracts named entities using spaCy en_core_web_trf (GPU).

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


def fetch_raw_text_for_window(subreddit, start_ts, end_ts, storage_options):
    """Fetch raw text from S3 for a specific subreddit + time range.

    Reads from the correct month partitions, filters by subreddit and time.
    Returns list of text strings.
    """
    start_dt = pd.Timestamp(start_ts, unit="s") if isinstance(start_ts, (int, float)) else pd.Timestamp(start_ts)
    end_dt = pd.Timestamp(end_ts, unit="s") if isinstance(end_ts, (int, float)) else pd.Timestamp(end_ts)

    # Determine which months to read
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
                    filters=[("subreddit", "==", subreddit)],
                )

                if len(df) == 0:
                    continue

                # Filter by time range
                df["ts"] = pd.to_datetime(df["created_utc"], unit="s")
                mask = (df["ts"] >= start_dt) & (df["ts"] <= end_dt)
                df = df[mask]

                # Extract text
                text_series = df[text_col].dropna().astype(str)
                # Filter out deleted/removed
                text_series = text_series[
                    ~text_series.isin(["[deleted]", "[removed]", ""])
                ]
                texts.extend(text_series.tolist())

            except Exception as e:
                logger.warning(f"Could not read {data_type} {year}-{month:02d} "
                               f"for r/{subreddit}: {e}")

    # Also include submission titles
    for year, month in months_needed:
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

    return texts


def extract_entities_batch(nlp, texts, batch_size=NER_BATCH_SIZE):
    """Run NER on a batch of texts using spacy.pipe for efficiency.

    Returns:
      entity_counts: Counter of (entity_text, entity_label) tuples
      cooccurrence: Counter of (ent1, ent2) pairs within same document
    """
    entity_counts = Counter()
    cooccurrence = Counter()
    doc_entity_lists = []

    # Truncate very long texts to avoid memory issues
    truncated = [t[:5000] if len(t) > 5000 else t for t in texts]

    n_batches = (len(truncated) + batch_size - 1) // batch_size
    logger.info(f"  Processing {len(truncated)} texts in {n_batches} batches "
                f"(batch_size={batch_size})")

    processed = 0
    for doc in nlp.pipe(truncated, batch_size=batch_size):
        # Extract entities of interest
        doc_entities = set()
        for ent in doc.ents:
            if ent.label_ in ENTITY_TYPES:
                # Normalize entity text
                ent_text = ent.text.strip()
                if len(ent_text) > 1:  # skip single-char entities
                    key = (ent_text, ent.label_)
                    entity_counts[key] += 1
                    doc_entities.add(key)

        # Co-occurrence within same document
        if len(doc_entities) >= 2:
            for pair in combinations(sorted(doc_entities), 2):
                cooccurrence[pair] += 1

        doc_entity_lists.append(list(doc_entities))
        processed += 1
        if processed % batch_size == 0:
            logger.info(f"    Processed {processed}/{len(truncated)} texts")

    return entity_counts, cooccurrence, doc_entity_lists


def process_anomaly_window(window_row, nlp, storage_options):
    """Process a single anomaly window: fetch text, run NER."""
    window_id = window_row.get("window_id", window_row.name)
    subreddit = window_row["subreddit"]
    start_ts = window_row["start_time"]
    end_ts = window_row["end_time"]

    logger.info(f"Window {window_id}: r/{subreddit} "
                f"({start_ts} to {end_ts})")

    # Fetch raw text
    texts = fetch_raw_text_for_window(subreddit, start_ts, end_ts, storage_options)
    logger.info(f"  Fetched {len(texts)} texts")

    if len(texts) == 0:
        return None, None

    # Cap at 100k texts to avoid excessive processing time
    if len(texts) > 100_000:
        logger.info(f"  Sampling 100k from {len(texts)} texts")
        rng = np.random.RandomState(42)
        indices = rng.choice(len(texts), 100_000, replace=False)
        texts = [texts[i] for i in indices]

    # Run NER
    entity_counts, cooccurrence, _ = extract_entities_batch(nlp, texts)

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

    # Load spaCy model
    nlp = load_spacy_model()

    # ── Process each anomaly window ─────────────────────────────────────────
    all_entity_rows = []
    all_cooc_rows = []
    n_windows = len(anomaly_windows)

    for idx, (_, row) in enumerate(tqdm(anomaly_windows.iterrows(),
                                         total=n_windows,
                                         desc="NER on anomaly windows")):
        window_id = row.get("window_id", idx)

        try:
            entity_counts, cooccurrence = process_anomaly_window(
                row, nlp, storage_options
            )
        except Exception as e:
            logger.error(f"Error processing window {window_id}: {e}")
            continue

        if entity_counts is None:
            continue

        # Build entity rows
        for (ent_text, ent_label), count in entity_counts.most_common(200):
            all_entity_rows.append({
                "window_id": window_id,
                "subreddit": row["subreddit"],
                "entity_text": ent_text,
                "entity_label": ent_label,
                "count": count,
            })

        # Build co-occurrence rows
        for (ent1, ent2), count in cooccurrence.most_common(100):
            all_cooc_rows.append({
                "window_id": window_id,
                "entity1_text": ent1[0],
                "entity1_label": ent1[1],
                "entity2_text": ent2[0],
                "entity2_label": ent2[1],
                "cooccurrence_count": count,
            })

        # Periodic checkpoint
        if (idx + 1) % 50 == 0:
            logger.info(f"  Checkpoint: {idx + 1}/{n_windows} windows processed, "
                        f"{len(all_entity_rows)} entity rows accumulated")

        gc.collect()

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
    logger.info(f"  Windows processed:    {n_windows}")
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
