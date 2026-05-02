"""Central project constants and paths."""

import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(PROJECT_DIR)

# ── S3 paths ──────────────────────────────────────────────────────────────
RAW_S3_BUCKET = os.environ.get("RAW_S3_BUCKET", "reddit-event-prediction-1776283460")
PROCESSED_S3_BUCKET = os.environ.get("PROCESSED_S3_BUCKET", RAW_S3_BUCKET)

# Backward-compatible alias used by existing stage imports.
S3_BUCKET = PROCESSED_S3_BUCKET

S3_BASE = os.environ.get("S3_BASE", f"s3://{RAW_S3_BUCKET}/reddit-data/parquet")
S3_COMMENTS = f"{S3_BASE}/comments"
S3_SUBMISSIONS = f"{S3_BASE}/submissions"
S3_INTERMEDIATE = os.environ.get(
    "S3_INTERMEDIATE",
    f"s3://{PROCESSED_S3_BUCKET}/reddit-data/intermediate",
)

# S3A versions (for PySpark)
S3A_COMMENTS = S3_COMMENTS.replace("s3://", "s3a://")
S3A_SUBMISSIONS = S3_SUBMISSIONS.replace("s3://", "s3a://")
S3A_INTERMEDIATE = S3_INTERMEDIATE.replace("s3://", "s3a://")

# ── Local paths ───────────────────────────────────────────────────────────
LOCAL_DATA = os.path.join(PROJECT_DIR, "data")
LOCAL_INTERMEDIATE = os.path.join(LOCAL_DATA, "intermediate")
LOCAL_GROUND_TRUTH = os.path.join(LOCAL_DATA, "ground_truth")
LOCAL_FIGURES = os.path.join(REPO_DIR, "Final-Project-Report", "figures", "report")

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
