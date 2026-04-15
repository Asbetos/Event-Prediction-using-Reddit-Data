#!/usr/bin/env python3
"""
Stage 8: GPU-Accelerated Topic Modeling with BERTopic (Q7)
============================================================
Collects text from all anomaly windows and performs topic modeling using
BERTopic with sentence-transformers embeddings, GPU-accelerated UMAP,
and HDBSCAN clustering.

Outputs:
  - topics.parquet: dominant topic per anomaly window with representative words

Usage: python stage8_topics_gpu.py
"""

import os
import sys
import time
import logging
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import s3fs
from tqdm import tqdm

# ── GPU imports with fallback ───────────────────────────────────────────────
output_exists = False
    try:
    import cudf
    GPU_AVAILABLE = True
    print("cuDF available")
except ImportError:
    GPU_AVAILABLE = False
    print("cuDF not available - using pandas")

output_exists = False
    try:
    import cuml
    CUML_AVAILABLE = True
    print("cuML available - GPU UMAP + HDBSCAN")
except ImportError:
    CUML_AVAILABLE = False
    print("cuML not available - using CPU UMAP + HDBSCAN")

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_DIR = os.environ.get("LOG_DIR", "/workspace/logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "stage8_topics.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import S3_BUCKET, S3_BASE, S3_INTERMEDIATE

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_TEXTS_PER_WINDOW = 5000
MAX_TOTAL_DOCS = 200_000  # BERTopic can handle this on A100


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
                           max_texts=MAX_TEXTS_PER_WINDOW):
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
        for data_type, text_col in [("comments", "body"), ("submissions", "selftext"),
                                     ("submissions", "title")]:
            output_exists = False
    try:
                cols = ["subreddit", "created_utc"]
                if text_col not in cols:
                    cols.append(text_col)
                path = f"{S3_BASE}/{data_type}/yyyy={year}/mm={month:02d}/"
                df = pd.read_parquet(
                    path,
                    columns=cols,
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
            except Exception:
                pass

    # Sample if too many
    if len(texts) > max_texts:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(texts), max_texts, replace=False)
        texts = [texts[i] for i in indices]

    return texts


def build_bertopic_model():
    """Build a BERTopic model with GPU-accelerated components where possible."""
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    # Embedding model (GPU via sentence-transformers + torch)
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")

    # UMAP: GPU via cuML or CPU fallback
    if CUML_AVAILABLE:
        from cuml.manifold import UMAP as cuUMAP
        logger.info("Using cuML GPU UMAP")
        umap_model = cuUMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
    else:
        from umap import UMAP
        logger.info("Using CPU UMAP")
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            low_memory=True,
        )

    # HDBSCAN: GPU via cuML or CPU fallback
    if CUML_AVAILABLE:
        from cuml.cluster import HDBSCAN as cuHDBSCAN
        logger.info("Using cuML GPU HDBSCAN")
        hdbscan_model = cuHDBSCAN(
            min_cluster_size=15,
            min_samples=5,
            gen_min_span_tree=True,
            prediction_data=True,
        )
    else:
        from hdbscan import HDBSCAN
        logger.info("Using CPU HDBSCAN")
        hdbscan_model = HDBSCAN(
            min_cluster_size=15,
            min_samples=5,
            gen_min_span_tree=True,
            prediction_data=True,
        )

    # Vectorizer for topic representation
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
    )

    # Build BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=10,
        verbose=True,
        calculate_probabilities=False,  # save memory
    )

    return topic_model, embedding_model


def main():
    logger.info("=" * 70)
    logger.info("Stage 8: GPU-Accelerated Topic Modeling with BERTopic")
    logger.info("=" * 70)
    t_start = time.time()

    storage_options = get_s3_storage_options()
    s3 = get_s3fs_client()

    # Check for existing output
    output_path = f"{S3_INTERMEDIATE}/topics.parquet"
    output_exists = False
    try:
        if s3.exists(output_path.replace("s3://", "")):
            logger.info("topics.parquet already exists on S3. Skipping.")
            logger.info("Delete the file to force re-run.")
            return
    except Exception:
        pass

    # Load anomaly windows
    anomaly_windows = read_anomaly_windows(storage_options)

    # ── Collect text from all anomaly windows ───────────────────────────────
    logger.info("Collecting text from all anomaly windows...")
    all_texts = []
    text_window_ids = []  # track which window each text belongs to
    text_subreddits = []

    for idx, (_, row) in enumerate(tqdm(anomaly_windows.iterrows(),
                                         total=len(anomaly_windows),
                                         desc="Fetching texts")):
        window_id = row.get("window_id", idx)
        subreddit = row["subreddit"]

        texts = fetch_texts_for_window(
            subreddit, row["start_time"], row["end_time"],
            storage_options, max_texts=MAX_TEXTS_PER_WINDOW,
        )

        for t in texts:
            all_texts.append(t)
            text_window_ids.append(window_id)
            text_subreddits.append(subreddit)

        if (idx + 1) % 25 == 0:
            logger.info(f"  {idx + 1}/{len(anomaly_windows)} windows, "
                        f"{len(all_texts)} total texts")

    logger.info(f"Total texts collected: {len(all_texts):,}")

    if len(all_texts) == 0:
        logger.error("No texts collected. Check anomaly windows and S3 data.")
        sys.exit(1)

    # Sample if total exceeds limit
    if len(all_texts) > MAX_TOTAL_DOCS:
        logger.info(f"Sampling {MAX_TOTAL_DOCS:,} from {len(all_texts):,} texts")
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_texts), MAX_TOTAL_DOCS, replace=False)
        all_texts = [all_texts[i] for i in indices]
        text_window_ids = [text_window_ids[i] for i in indices]
        text_subreddits = [text_subreddits[i] for i in indices]

    # ── Build and fit BERTopic ──────────────────────────────────────────────
    logger.info("Building BERTopic model...")
    topic_model, embedding_model = build_bertopic_model()

    logger.info("Computing embeddings...")
    t_embed = time.time()
    embeddings = embedding_model.encode(
        all_texts,
        batch_size=512,
        show_progress_bar=True,
        device="cuda",
    )
    logger.info(f"  Embeddings computed in {time.time() - t_embed:.1f}s "
                f"(shape: {embeddings.shape})")

    logger.info("Fitting BERTopic model...")
    t_fit = time.time()
    topics, probs = topic_model.fit_transform(all_texts, embeddings)
    logger.info(f"  BERTopic fitted in {time.time() - t_fit:.1f}s")

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)
    logger.info(f"  Found {n_topics} topics, {n_outliers} outliers")

    # ── Extract topic info ──────────────────────────────────────────────────
    topic_info = topic_model.get_topic_info()
    logger.info(f"Topic info:\n{topic_info.head(20).to_string()}")

    # ── Assign dominant topic to each anomaly window ────────────────────────
    logger.info("Assigning dominant topics to anomaly windows...")

    # Build mapping: window_id -> list of topic assignments
    window_topics = {}
    for wid, topic_id in zip(text_window_ids, topics):
        if wid not in window_topics:
            window_topics[wid] = []
        window_topics[wid].append(topic_id)

    topic_rows = []
    for idx, (_, row) in enumerate(anomaly_windows.iterrows()):
        window_id = row.get("window_id", idx)
        subreddit = row["subreddit"]

        if window_id not in window_topics:
            topic_rows.append({
                "window_id": window_id,
                "subreddit": subreddit,
                "dominant_topic": -1,
                "topic_words": "",
                "n_topics_present": 0,
                "topic_distribution": "{}",
            })
            continue

        wt = window_topics[window_id]

        # Count topic occurrences
        from collections import Counter
        topic_counts = Counter(wt)

        # Dominant topic (excluding outlier topic -1 if possible)
        non_outlier = {k: v for k, v in topic_counts.items() if k != -1}
        if non_outlier:
            dominant = max(non_outlier, key=non_outlier.get)
        else:
            dominant = -1

        # Get topic words
        topic_words = ""
        if dominant != -1:
            output_exists = False
    try:
                words_scores = topic_model.get_topic(dominant)
                topic_words = ", ".join([w for w, _ in words_scores[:10]])
            except Exception:
                pass

        # Topic distribution
        total = sum(topic_counts.values())
        dist = {str(k): round(v / total, 3) for k, v in topic_counts.most_common(5)}

        topic_rows.append({
            "window_id": window_id,
            "subreddit": subreddit,
            "dominant_topic": dominant,
            "topic_words": topic_words,
            "n_topics_present": len(non_outlier),
            "topic_distribution": str(dist),
        })

    topics_df = pd.DataFrame(topic_rows)

    # ── Also save full topic info ───────────────────────────────────────────
    topic_details = []
    for topic_id in topic_info["Topic"]:
        if topic_id == -1:
            continue
        output_exists = False
    try:
            words_scores = topic_model.get_topic(topic_id)
            detail = {
                "topic_id": topic_id,
                "count": int(topic_info[topic_info["Topic"] == topic_id]["Count"].iloc[0]),
                "representative_words": ", ".join([w for w, _ in words_scores[:10]]),
            }
            topic_details.append(detail)
        except Exception:
            pass

    if topic_details:
        topic_details_df = pd.DataFrame(topic_details)
        details_path = f"{S3_INTERMEDIATE}/topic_details.parquet"
        topic_details_df.to_parquet(
            details_path, index=False, storage_options=storage_options
        )
        logger.info(f"Topic details saved to {details_path}")

    # ── Write output ────────────────────────────────────────────────────────
    logger.info(f"Writing topics.parquet ({len(topics_df)} rows)...")
    topics_df.to_parquet(output_path, index=False, storage_options=storage_options)

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("Stage 8 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time:           {elapsed / 60:.1f} minutes")
    logger.info(f"  Documents processed:  {len(all_texts):,}")
    logger.info(f"  Topics found:         {n_topics}")
    logger.info(f"  Outlier documents:    {n_outliers:,} "
                f"({100 * n_outliers / len(all_texts):.1f}%)")
    logger.info(f"  Windows with topics:  "
                f"{(topics_df['dominant_topic'] != -1).sum()}/{len(topics_df)}")

    # Print top 5 topics
    if topic_details:
        logger.info("  Top topics:")
        for row in sorted(topic_details, key=lambda x: x["count"], reverse=True)[:5]:
            logger.info(f"    Topic {row['topic_id']} ({row['count']} docs): "
                        f"{row['representative_words'][:80]}")

    logger.info(f"  Output at:            {output_path}")


if __name__ == "__main__":
    main()
