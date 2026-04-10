#!/usr/bin/env python3
"""
Stage 2 -- Baseline Computation & Anomaly Detection (Q1)
=========================================================
Reads hourly_counts.parquet, computes 7-day rolling statistics per subreddit,
flags anomalous hours via z-score, and merges consecutive anomalies into
contiguous windows.

Outputs:
    anomaly_windows.parquet   (local intermediate)
    Figures: zscore_distribution.png, monthly_anomaly_counts.png
"""

import os, sys, logging, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config.spark_config import create_spark_session
from config.settings import (
    ZSCORE_THRESHOLD,
    ROLLING_WINDOW_HOURS,
    ANOMALY_MERGE_GAP_HOURS,
)
from utils.spark_utils import read_intermediate, write_intermediate
from utils.viz_utils import save_fig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("stage2")

# ===========================================================================
# Main
# ===========================================================================

def main():
    t0 = time.time()
    spark = create_spark_session(app_name="Stage2_AnomalyDetection")
    log.info("Spark session created.")

    # ----- 1. Read hourly counts -----------------------------------------
    try:
        hourly = read_intermediate(spark, "hourly_counts.parquet")
    except Exception as exc:
        log.error(
            "Could not read hourly_counts.parquet. "
            "Run Stage 1 first to produce intermediate data.\n%s", exc
        )
        spark.stop()
        sys.exit(1)

    row_count = hourly.count()
    log.info("hourly_counts.parquet: %s rows loaded.", f"{row_count:,}")

    # ----- 2. Rolling baseline (7-day window) ----------------------------
    # rangeBetween uses the *column value* difference in seconds for
    # TimestampType columns that Spark stores as epoch seconds internally.
    seconds_in_window = ROLLING_WINDOW_HOURS * 3600  # 168 h * 3600

    hourly = hourly.withColumn(
        "hour_bucket_seconds",
        F.col("hour_bucket").cast("long"),
    )

    w_rolling = (
        Window.partitionBy("subreddit")
        .orderBy("hour_bucket_seconds")
        .rangeBetween(-seconds_in_window, -1)
    )

    hourly = (
        hourly
        .withColumn("rolling_mean", F.avg("post_count").over(w_rolling))
        .withColumn("rolling_std",  F.stddev_pop("post_count").over(w_rolling))
    )

    # Z-score (guard against zero std)
    hourly = hourly.withColumn(
        "z_score",
        F.when(
            (F.col("rolling_std").isNotNull()) & (F.col("rolling_std") > 0),
            (F.col("post_count") - F.col("rolling_mean")) / F.col("rolling_std"),
        ).otherwise(0.0),
    )

    log.info("Rolling statistics and z-scores computed.")

    # ----- 3. Flag anomalous hours ---------------------------------------
    anomalies = hourly.filter(F.col("z_score") > ZSCORE_THRESHOLD)
    anomaly_count = anomalies.count()
    log.info("Anomalous hours (z > %.1f): %s", ZSCORE_THRESHOLD, f"{anomaly_count:,}")

    if anomaly_count == 0:
        log.warning("No anomalies detected -- nothing to merge. Exiting.")
        spark.stop()
        return

    # ----- 4. Merge consecutive anomalous hours into windows -------------
    w_lag = Window.partitionBy("subreddit").orderBy("hour_bucket")

    merge_gap_seconds = ANOMALY_MERGE_GAP_HOURS * 3600

    anomalies = anomalies.withColumn(
        "prev_hour_bucket", F.lag("hour_bucket").over(w_lag)
    )

    anomalies = anomalies.withColumn(
        "gap_seconds",
        F.when(
            F.col("prev_hour_bucket").isNotNull(),
            F.col("hour_bucket").cast("long") - F.col("prev_hour_bucket").cast("long"),
        ).otherwise(F.lit(merge_gap_seconds + 1).cast("long")),
    )

    # Mark the start of a new window when gap exceeds threshold
    anomalies = anomalies.withColumn(
        "new_window_flag",
        F.when(F.col("gap_seconds") > merge_gap_seconds, 1).otherwise(0),
    )

    # Cumulative sum to generate window_id per subreddit
    w_cumsum = (
        Window.partitionBy("subreddit")
        .orderBy("hour_bucket")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    anomalies = anomalies.withColumn(
        "window_id",
        F.concat_ws(
            "_",
            F.col("subreddit"),
            F.sum("new_window_flag").over(w_cumsum).cast("string"),
        ),
    )

    # ----- 5. Aggregate each anomaly window ------------------------------
    anomaly_windows = (
        anomalies.groupBy("subreddit", "window_id")
        .agg(
            F.min("hour_bucket").alias("window_start"),
            F.max("hour_bucket").alias("window_end"),
            F.max("z_score").alias("peak_z_score"),
            F.avg("z_score").alias("mean_z_score"),
            F.count("*").alias("anomaly_hours"),
            F.max("post_count").alias("peak_post_count"),
            F.avg("post_count").alias("mean_post_count"),
        )
        .withColumn(
            "duration_hours",
            (F.col("window_end").cast("long") - F.col("window_start").cast("long")) / 3600 + 1,
        )
    )

    # ----- 6. Write output -----------------------------------------------
    write_intermediate(anomaly_windows, "anomaly_windows.parquet")

    window_count = anomaly_windows.count()
    log.info("Anomaly windows written: %s", f"{window_count:,}")

    # =====================================================================
    # Summaries & Visualizations
    # =====================================================================
    # --- Top-20 anomalies by peak z-score --------------------------------
    top20_df = (
        anomaly_windows
        .orderBy(F.col("peak_z_score").desc())
        .limit(20)
        .select("subreddit", "window_start", "window_end",
                "peak_z_score", "mean_z_score", "duration_hours",
                "anomaly_hours")
    )
    top20_pd = top20_df.toPandas()

    print("\n" + "=" * 90)
    print("  TOP 20 ANOMALY WINDOWS BY PEAK Z-SCORE")
    print("=" * 90)
    print(top20_pd.to_string(index=False))
    print("=" * 90 + "\n")

    # --- 1. Z-score distribution histogram -------------------------------
    log.info("Generating z-score distribution histogram ...")

    zscore_sample = (
        hourly
        .filter(F.col("z_score").isNotNull())
        .select("z_score")
        .sample(fraction=min(1.0, 500_000 / max(row_count, 1)))
        .toPandas()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(zscore_sample["z_score"], bins=200, edgecolor="none", alpha=0.8,
            color="#3498DB")
    ax.axvline(ZSCORE_THRESHOLD, color="#E74C3C", ls="--", lw=2,
               label=f"Threshold ({ZSCORE_THRESHOLD})")
    ax.set_xlabel("Z-Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Hourly Post-Count Z-Scores")
    ax.set_xlim(-5, 20)
    ax.legend()
    save_fig(fig, "zscore_distribution.png")

    # --- 2. Monthly anomaly count bar chart ------------------------------
    log.info("Generating monthly anomaly count chart ...")

    monthly = (
        anomaly_windows
        .withColumn("month", F.date_format("window_start", "yyyy-MM"))
        .groupBy("month")
        .agg(F.count("*").alias("anomaly_count"))
        .orderBy("month")
        .toPandas()
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(monthly["month"], monthly["anomaly_count"], color="#2ECC71",
           edgecolor="white")
    ax.set_xlabel("Month")
    ax.set_ylabel("Anomaly Windows")
    ax.set_title("Number of Anomaly Windows per Month")
    plt.xticks(rotation=45, ha="right")
    save_fig(fig, "monthly_anomaly_counts.png")

    # ----- Summary stats -------------------------------------------------
    stats = anomaly_windows.agg(
        F.count("*").alias("total_windows"),
        F.countDistinct("subreddit").alias("unique_subreddits"),
        F.avg("duration_hours").alias("avg_duration_h"),
        F.max("peak_z_score").alias("max_peak_z"),
        F.avg("peak_z_score").alias("avg_peak_z"),
    ).toPandas().iloc[0]

    print("\n--- Stage 2 Summary ---")
    print(f"  Total anomaly windows : {int(stats['total_windows']):,}")
    print(f"  Unique subreddits     : {int(stats['unique_subreddits']):,}")
    print(f"  Avg duration (hours)  : {stats['avg_duration_h']:.1f}")
    print(f"  Max peak z-score      : {stats['max_peak_z']:.2f}")
    print(f"  Avg peak z-score      : {stats['avg_peak_z']:.2f}")
    print(f"  Elapsed time          : {time.time() - t0:.1f}s")

    spark.stop()
    log.info("Stage 2 complete.")


if __name__ == "__main__":
    main()
