#!/usr/bin/env python3
"""
Stage 5 -- Temporal Patterns Analysis (Q4)
===========================================
Extracts hour-of-day and day-of-week patterns from hourly activity and
anomaly data, joins with ground-truth events to map category-to-time
distributions, and compares weekday vs weekend propagation speed.

Outputs:
    temporal_patterns.parquet   (local intermediate)
    Figures: anomaly_heatmap_dow_hour.png,
             anomaly_polar_hour.png,
             event_category_by_dow.png
"""

import os, sys, logging, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType, ArrayType,
)

from config.spark_config import create_spark_session
from config.settings import LOCAL_GROUND_TRUTH
from utils.spark_utils import read_intermediate, write_intermediate
from utils.viz_utils import save_fig, CATEGORY_COLORS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("stage5")

DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_ground_truth(spark):
    """Load the ground truth events CSV and explode relevant_subreddits."""
    gt_path = os.path.join(LOCAL_GROUND_TRUTH, "events.csv")
    if not os.path.exists(gt_path):
        log.warning("Ground truth file not found at %s", gt_path)
        return None

    gt = (
        spark.read.csv(gt_path, header=True, inferSchema=True)
        .withColumn("date", F.to_date("date", "yyyy-MM-dd"))
        .withColumn("subreddits_arr", F.split(F.col("relevant_subreddits"), "\\|"))
    )
    # Explode so each subreddit gets its own row
    gt_exploded = gt.select(
        "event_id", "date", "name", "category",
        F.explode("subreddits_arr").alias("subreddit"),
    )
    return gt_exploded


def main():
    t0 = time.time()
    spark = create_spark_session(app_name="Stage5_Temporal")
    log.info("Spark session created.")

    # ----- 1. Read inputs ------------------------------------------------
    try:
        hourly = read_intermediate(spark, "hourly_counts.parquet")
        anomaly_windows = read_intermediate(spark, "anomaly_windows.parquet")
    except Exception as exc:
        log.error(
            "Could not read required intermediate data. "
            "Run Stages 1-2 first.\n%s", exc
        )
        spark.stop()
        sys.exit(1)

    log.info("Hourly counts: %s rows", f"{hourly.count():,}")
    log.info("Anomaly windows: %s rows", f"{anomaly_windows.count():,}")

    # ----- 2. Temporal features on hourly data ---------------------------
    hourly = hourly.withColumn(
        "hour_of_day", F.hour("hour_bucket")
    ).withColumn(
        "day_of_week", F.dayofweek("hour_bucket") - 2
        # Spark dayofweek: 1=Sun..7=Sat -> shift to 0=Mon..6=Sun
    ).withColumn(
        "day_of_week",
        F.when(F.col("day_of_week") < 0, F.col("day_of_week") + 7)
        .otherwise(F.col("day_of_week")),
    )

    # Baseline temporal profile: average activity by (hour, dow)
    baseline_profile = (
        hourly.groupBy("hour_of_day", "day_of_week")
        .agg(
            F.avg("post_count").alias("avg_post_count"),
            F.avg("unique_authors").alias("avg_unique_authors"),
            F.sum("post_count").alias("total_post_count"),
            F.count("*").alias("num_observations"),
        )
    )

    log.info("Baseline temporal profile computed.")

    # ----- 3. Anomaly temporal profile -----------------------------------
    anomaly_windows = anomaly_windows.withColumn(
        "hour_of_day", F.hour("window_start")
    ).withColumn(
        "day_of_week", F.dayofweek("window_start") - 2
    ).withColumn(
        "day_of_week",
        F.when(F.col("day_of_week") < 0, F.col("day_of_week") + 7)
        .otherwise(F.col("day_of_week")),
    )

    anomaly_profile = (
        anomaly_windows.groupBy("hour_of_day", "day_of_week")
        .agg(
            F.count("*").alias("anomaly_count"),
            F.avg("peak_z_score").alias("avg_peak_z"),
        )
    )

    log.info("Anomaly temporal profile computed.")

    # ----- 4. Join with ground truth events ------------------------------
    gt = load_ground_truth(spark)

    gt_temporal = None
    if gt is not None:
        gt = gt.withColumn(
            "day_of_week", F.dayofweek("date") - 2
        ).withColumn(
            "day_of_week",
            F.when(F.col("day_of_week") < 0, F.col("day_of_week") + 7)
            .otherwise(F.col("day_of_week")),
        )

        gt_temporal = (
            gt.groupBy("category", "day_of_week")
            .agg(F.countDistinct("event_id").alias("event_count"))
        )
        log.info("Ground truth temporal join done.")
    else:
        log.warning("Skipping ground truth join (file not found).")

    # ----- 5. Weekday vs weekend anomaly comparison ----------------------
    anomaly_windows = anomaly_windows.withColumn(
        "is_weekend",
        F.when(F.col("day_of_week").isin(5, 6), True).otherwise(False),
    )

    weekday_weekend = (
        anomaly_windows.groupBy("is_weekend")
        .agg(
            F.count("*").alias("anomaly_count"),
            F.avg("peak_z_score").alias("avg_peak_z"),
            F.avg("duration_hours").alias("avg_duration_h"),
        )
    )

    ww_pd = weekday_weekend.toPandas()
    print("\n  Weekday vs Weekend Anomalies:")
    print(ww_pd.to_string(index=False))

    # Try to join propagation data for speed comparison
    try:
        prop_events = read_intermediate(spark, "propagation_events.parquet")
        prop_events = prop_events.withColumn(
            "day_of_week", F.dayofweek("first_detection_time") - 2
        ).withColumn(
            "day_of_week",
            F.when(F.col("day_of_week") < 0, F.col("day_of_week") + 7)
            .otherwise(F.col("day_of_week")),
        ).withColumn(
            "is_weekend",
            F.when(F.col("day_of_week").isin(5, 6), True).otherwise(False),
        )

        prop_speed = (
            prop_events.groupBy("is_weekend")
            .agg(
                F.avg("total_duration_hours").alias("avg_propagation_hours"),
                F.avg("num_subreddits").alias("avg_num_subreddits"),
            )
        )
        print("\n  Weekday vs Weekend Propagation Speed:")
        print(prop_speed.toPandas().to_string(index=False))
    except Exception:
        log.info("propagation_events.parquet not available; skipping speed comparison.")

    # ----- 6. Combine & write output -------------------------------------
    temporal_out = (
        baseline_profile.join(
            anomaly_profile, on=["hour_of_day", "day_of_week"], how="left"
        )
        .fillna(0, subset=["anomaly_count"])
    )

    write_intermediate(temporal_out, "temporal_patterns.parquet")

    # =====================================================================
    # Visualizations
    # =====================================================================
    # --- 1. Heatmap: day_of_week x hour_of_day (anomaly count) -----------
    log.info("Generating anomaly heatmap ...")
    anom_prof_pd = anomaly_profile.toPandas()

    heatmap = np.zeros((7, 24))
    for _, row in anom_prof_pd.iterrows():
        dow = int(row["day_of_week"])
        hod = int(row["hour_of_day"])
        if 0 <= dow < 7 and 0 <= hod < 24:
            heatmap[dow, hod] = row["anomaly_count"]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(7))
    ax.set_yticklabels(DOW_LABELS)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Day of Week")
    ax.set_title("Anomaly Count by Day of Week and Hour of Day")
    fig.colorbar(im, ax=ax, label="Anomaly Count")
    save_fig(fig, "anomaly_heatmap_dow_hour.png")

    # --- 2. Polar / radial chart of anomaly frequency by hour ------------
    log.info("Generating polar anomaly chart ...")
    hourly_anomalies = anom_prof_pd.groupby("hour_of_day")["anomaly_count"].sum()
    hourly_anomalies = hourly_anomalies.reindex(range(24), fill_value=0)

    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    values = hourly_anomalies.values.astype(float)
    # Close the circle
    angles_closed = np.append(angles, angles[0])
    values_closed = np.append(values, values[0])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.plot(angles_closed, values_closed, color="#E74C3C", lw=2)
    ax.fill(angles_closed, values_closed, alpha=0.25, color="#E74C3C")
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], fontsize=8)
    ax.set_title("Anomaly Frequency by Hour of Day (UTC)", pad=20, fontsize=13)
    save_fig(fig, "anomaly_polar_hour.png")

    # --- 3. Event category distribution by day_of_week (stacked bar) -----
    if gt_temporal is not None:
        log.info("Generating event category by day-of-week chart ...")
        gt_pd = gt_temporal.toPandas()

        categories = gt_pd["category"].unique()
        dow_range = range(7)

        fig, ax = plt.subplots(figsize=(12, 7))
        bottom = np.zeros(7)
        for cat in sorted(categories):
            cat_data = gt_pd[gt_pd["category"] == cat]
            counts = np.zeros(7)
            for _, row in cat_data.iterrows():
                dow = int(row["day_of_week"])
                if 0 <= dow < 7:
                    counts[dow] = row["event_count"]
            color = CATEGORY_COLORS.get(cat, "#888888")
            ax.bar(dow_range, counts, bottom=bottom, label=cat, color=color,
                   edgecolor="white", width=0.7)
            bottom += counts

        ax.set_xticks(range(7))
        ax.set_xticklabels(DOW_LABELS)
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Event Count")
        ax.set_title("Ground-Truth Event Categories by Day of Week")
        ax.legend(loc="upper right")
        save_fig(fig, "event_category_by_dow.png")
    else:
        log.info("Skipping event category chart (no ground truth).")

    # ----- Summary stats -------------------------------------------------
    temporal_pd = temporal_out.toPandas()
    peak_slot = temporal_pd.loc[temporal_pd["anomaly_count"].idxmax()] if not temporal_pd.empty else None

    print("\n--- Stage 5 Summary ---")
    print(f"  Temporal pattern rows       : {len(temporal_pd):,}")
    if peak_slot is not None:
        dow_label = DOW_LABELS[int(peak_slot["day_of_week"])] if int(peak_slot["day_of_week"]) < 7 else "?"
        print(f"  Peak anomaly slot           : {dow_label} {int(peak_slot['hour_of_day']):02d}:00 UTC "
              f"({int(peak_slot['anomaly_count'])} anomalies)")
    print(f"  Total anomalies in profile  : {int(temporal_pd['anomaly_count'].sum()):,}")
    if gt is not None:
        print(f"  Ground truth events loaded  : {gt.select('event_id').distinct().count()}")
    print(f"  Elapsed time                : {time.time() - t0:.1f}s")

    spark.stop()
    log.info("Stage 5 complete.")


if __name__ == "__main__":
    main()
