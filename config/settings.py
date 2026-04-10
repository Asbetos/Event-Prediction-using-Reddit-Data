"""Central project constants and paths."""

import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── S3 paths ──────────────────────────────────────────────────────────────
S3_BUCKET = "ven-bda-s3-v2"
S3_BASE = f"s3://{S3_BUCKET}/reddit-data/parquet"
S3_COMMENTS = f"{S3_BASE}/comments"
S3_SUBMISSIONS = f"{S3_BASE}/submissions"
S3_INTERMEDIATE = f"s3://{S3_BUCKET}/reddit-data/intermediate"

# S3A versions (for PySpark)
S3A_COMMENTS = S3_COMMENTS.replace("s3://", "s3a://")
S3A_SUBMISSIONS = S3_SUBMISSIONS.replace("s3://", "s3a://")
S3A_INTERMEDIATE = S3_INTERMEDIATE.replace("s3://", "s3a://")

# ── Local paths ───────────────────────────────────────────────────────────
LOCAL_DATA = os.path.join(PROJECT_DIR, "data")
LOCAL_INTERMEDIATE = os.path.join(LOCAL_DATA, "intermediate")
LOCAL_GROUND_TRUTH = os.path.join(LOCAL_DATA, "ground_truth")
LOCAL_FIGURES = os.path.join(PROJECT_DIR, "website", "figures")

# ── Date range ────────────────────────────────────────────────────────────
MONTHS = [
    (2023, m) for m in range(6, 13)
] + [
    (2024, m) for m in range(1, 8)
]  # June 2023 through July 2024

# ── Anomaly detection parameters ─────────────────────────────────────────
ZSCORE_THRESHOLD = 3.0
ROLLING_WINDOW_HOURS = 168          # 7 days
ANOMALY_MERGE_GAP_HOURS = 6         # merge anomalies within this gap
TOP_N_SUBREDDITS = 500

# ── Event categories ─────────────────────────────────────────────────────
EVENT_CATEGORIES = [
    "breaking_news",
    "controversy",
    "product_launch",
    "disaster",
    "meme_viral",
]
