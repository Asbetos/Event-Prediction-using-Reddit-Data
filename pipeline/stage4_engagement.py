#!/usr/bin/env python3
"""
Stage 4 -- Spike Shape Classification & Engagement Analysis (Q3)
=================================================================
For each anomaly window, extracts the surrounding time series, classifies
the spike shape (sharp spike, sustained plateau, double peak, slow burn),
and correlates spike magnitude with post-spike engagement metrics.

Outputs:
    spike_profiles.parquet   (local intermediate)
    Figures: spike_shape_examples.png, spike_magnitude_vs_engagement.png,
             spike_shape_distribution.png
"""

import os, sys, logging, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config.spark_config import create_spark_session
from utils.spark_utils import read_intermediate, write_intermediate
from utils.viz_utils import save_fig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("stage4")

# Time-series extraction window
PRE_HOURS = 24
POST_HOURS = 72


def classify_spike_shape(ts_values):
    """Classify a spike shape from a normalized time-series array.

    Parameters
    ----------
    ts_values : list[float]
        Normalized post counts (0-1 scale), length = PRE_HOURS + POST_HOURS + 1.
        Index 0 = -24h ... index PRE_HOURS = anomaly start ... last = +72h.

    Returns
    -------
    str : one of "sharp_spike", "sustained_plateau", "double_peak", "slow_burn"
    """
    arr = np.array(ts_values, dtype=float)
    if len(arr) == 0 or np.nanmax(arr) == 0:
        return "sharp_spike"

    # Normalize to 0-1
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    if vmax - vmin < 1e-9:
        return "sustained_plateau"
    norm = (arr - vmin) / (vmax - vmin)

    peak_idx = int(np.nanargmax(norm))
    baseline = float(np.nanmean(norm[:PRE_HOURS])) if PRE_HOURS > 0 else 0.0

    # Time-to-peak from anomaly start (index PRE_HOURS)
    time_to_peak = max(peak_idx - PRE_HOURS, 0)

    # Time-to-half-decay: hours after peak until value drops below half of peak
    peak_val = norm[peak_idx]
    half_peak = peak_val / 2
    time_to_half_decay = None
    for i in range(peak_idx + 1, len(norm)):
        if norm[i] <= half_peak:
            time_to_half_decay = i - peak_idx
            break

    # Sustained above 2x baseline
    threshold_2x = max(baseline * 2, 0.3)  # floor at 0.3 for normalization edge cases
    sustained_hours = 0
    for v in norm[PRE_HOURS:]:
        if v >= threshold_2x:
            sustained_hours += 1

    # Double peak detection: find local maxima with >20% dip between them
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(norm, distance=3, prominence=0.2)
    # Check for dip between peaks
    has_double_peak = False
    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            valley = np.min(norm[peaks[i]:peaks[i + 1] + 1])
            avg_peak = (norm[peaks[i]] + norm[peaks[i + 1]]) / 2
            if avg_peak > 0 and (avg_peak - valley) / avg_peak > 0.2:
                has_double_peak = True
                break

    # Classification hierarchy
    if has_double_peak:
        return "double_peak"
    if time_to_peak < 4 and time_to_half_decay is not None and time_to_half_decay < 8:
        return "sharp_spike"
    if sustained_hours > 24:
        return "sustained_plateau"
    if time_to_peak > 12:
        return "slow_burn"
    # Default
    if time_to_half_decay is not None and time_to_half_decay < 12:
        return "sharp_spike"
    return "sustained_plateau"


def main():
    t0 = time.time()
    spark = create_spark_session(app_name="Stage4_Engagement")
    log.info("Spark session created.")

    # ----- 1. Read inputs ------------------------------------------------
    try:
        anomaly_windows = read_intermediate(spark, "anomaly_windows.parquet")
        hourly = read_intermediate(spark, "hourly_counts.parquet")
    except Exception as exc:
        log.error(
            "Could not read required intermediate data. "
            "Run Stages 1-2 first.\n%s", exc
        )
        spark.stop()
        sys.exit(1)

    aw_count = anomaly_windows.count()
    log.info("Anomaly windows: %s", f"{aw_count:,}")

    if aw_count == 0:
        log.warning("No anomaly windows to process. Exiting.")
        spark.stop()
        return

    # ----- 2. Extract time series per anomaly window ---------------------
    # Work in pandas for per-window time series extraction (aw is small).
    aw_pd = anomaly_windows.select(
        "window_id", "subreddit", "window_start", "window_end",
        "peak_z_score", "mean_z_score", "duration_hours",
    ).toPandas()

    hourly_pd = hourly.select(
        "subreddit", "hour_bucket", "post_count", "unique_authors", "mean_score",
    ).toPandas()
    hourly_pd["hour_bucket"] = pd.to_datetime(hourly_pd["hour_bucket"])
    hourly_pd = hourly_pd.sort_values(["subreddit", "hour_bucket"])

    log.info("Extracting time series for %d anomaly windows ...", len(aw_pd))

    profiles = []
    ts_examples = {
        "sharp_spike": None,
        "sustained_plateau": None,
        "double_peak": None,
        "slow_burn": None,
    }

    for _, aw_row in aw_pd.iterrows():
        sub = aw_row["subreddit"]
        ws = pd.Timestamp(aw_row["window_start"])
        we = pd.Timestamp(aw_row["window_end"])

        extract_start = ws - pd.Timedelta(hours=PRE_HOURS)
        extract_end = we + pd.Timedelta(hours=POST_HOURS)

        mask = (
            (hourly_pd["subreddit"] == sub)
            & (hourly_pd["hour_bucket"] >= extract_start)
            & (hourly_pd["hour_bucket"] <= extract_end)
        )
        ts = hourly_pd.loc[mask].sort_values("hour_bucket")

        if ts.empty:
            continue

        ts_values = ts["post_count"].values.tolist()
        spike_shape = classify_spike_shape(ts_values)

        # Post-spike engagement (after window_end)
        post_mask = (
            (hourly_pd["subreddit"] == sub)
            & (hourly_pd["hour_bucket"] > we)
            & (hourly_pd["hour_bucket"] <= we + pd.Timedelta(hours=POST_HOURS))
        )
        post_ts = hourly_pd.loc[post_mask]
        post_spike_avg_score = float(post_ts["mean_score"].mean()) if not post_ts.empty else 0.0
        post_spike_unique_authors = float(post_ts["unique_authors"].mean()) if not post_ts.empty else 0.0

        # Pre-spike baseline
        pre_mask = (
            (hourly_pd["subreddit"] == sub)
            & (hourly_pd["hour_bucket"] >= extract_start)
            & (hourly_pd["hour_bucket"] < ws)
        )
        pre_ts = hourly_pd.loc[pre_mask]
        baseline_avg_score = float(pre_ts["mean_score"].mean()) if not pre_ts.empty else 0.0

        profiles.append({
            "window_id": aw_row["window_id"],
            "subreddit": sub,
            "window_start": aw_row["window_start"],
            "window_end": aw_row["window_end"],
            "peak_z_score": float(aw_row["peak_z_score"]),
            "mean_z_score": float(aw_row["mean_z_score"]),
            "duration_hours": float(aw_row["duration_hours"]),
            "spike_shape": spike_shape,
            "post_spike_avg_score": post_spike_avg_score,
            "post_spike_unique_authors": post_spike_unique_authors,
            "baseline_avg_score": baseline_avg_score,
            "engagement_ratio": (
                post_spike_avg_score / baseline_avg_score
                if baseline_avg_score > 0 else 0.0
            ),
        })

        # Store example time series for visualization
        if ts_examples[spike_shape] is None:
            ts_examples[spike_shape] = {
                "subreddit": sub,
                "window_start": ws,
                "hours": [
                    (t - ws).total_seconds() / 3600
                    for t in ts["hour_bucket"]
                ],
                "values": ts_values,
                "peak_z": float(aw_row["peak_z_score"]),
            }

    profiles_pd = pd.DataFrame(profiles)
    log.info("Spike profiles computed: %d", len(profiles_pd))

    if profiles_pd.empty:
        log.warning("No spike profiles generated. Exiting.")
        spark.stop()
        return

    # ----- 3. Write output -----------------------------------------------
    profiles_sdf = spark.createDataFrame(profiles_pd)
    write_intermediate(profiles_sdf, "spike_profiles.parquet")

    # ----- 4. Correlation analysis ---------------------------------------
    valid = profiles_pd.dropna(subset=["peak_z_score", "post_spike_avg_score"])
    valid = valid[valid["post_spike_avg_score"] > 0]

    if len(valid) > 5:
        pearson_r, pearson_p = scipy_stats.pearsonr(
            valid["peak_z_score"], valid["post_spike_avg_score"]
        )
        spearman_r, spearman_p = scipy_stats.spearmanr(
            valid["peak_z_score"], valid["post_spike_avg_score"]
        )
        print(f"\n  Pearson  r={pearson_r:.4f}  p={pearson_p:.4e}")
        print(f"  Spearman r={spearman_r:.4f}  p={spearman_p:.4e}")
    else:
        pearson_r = spearman_r = float("nan")
        log.warning("Not enough data points for correlation analysis.")

    # =====================================================================
    # Visualizations
    # =====================================================================
    # --- 1. Small multiples: example time series per spike shape ---------
    log.info("Generating spike shape examples ...")
    shape_names = ["sharp_spike", "sustained_plateau", "double_peak", "slow_burn"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, shape in enumerate(shape_names):
        ax = axes[idx]
        ex = ts_examples.get(shape)
        if ex is not None:
            ax.plot(ex["hours"], ex["values"], color="#3498DB", lw=1.5)
            ax.axvline(0, color="#E74C3C", ls="--", alpha=0.7, label="Window start")
            ax.fill_between(ex["hours"], ex["values"], alpha=0.15, color="#3498DB")
            ax.set_title(
                f'{shape.replace("_", " ").title()}\n'
                f'r/{ex["subreddit"]} (z={ex["peak_z"]:.1f})',
                fontsize=11,
            )
        else:
            ax.text(0.5, 0.5, "No example\navailable",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(shape.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Hours relative to window start")
        ax.set_ylabel("Post count")

    fig.suptitle("Spike Shape Categories -- Example Time Series", fontsize=14)
    save_fig(fig, "spike_shape_examples.png")

    # --- 2. Scatter: magnitude vs engagement with regression line --------
    log.info("Generating magnitude vs engagement scatter ...")
    if len(valid) > 2:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(
            valid["peak_z_score"], valid["post_spike_avg_score"],
            alpha=0.4, s=20, c="#3498DB", edgecolors="white", linewidth=0.3,
        )
        # Regression line
        z = np.polyfit(valid["peak_z_score"], valid["post_spike_avg_score"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid["peak_z_score"].min(), valid["peak_z_score"].max(), 100)
        ax.plot(x_line, p(x_line), color="#E74C3C", lw=2,
                label=f"OLS (Pearson r={pearson_r:.3f})")
        ax.set_xlabel("Peak Z-Score (spike magnitude)")
        ax.set_ylabel("Post-Spike Average Score (engagement)")
        ax.set_title("Spike Magnitude vs Post-Spike Engagement")
        ax.legend()
        save_fig(fig, "spike_magnitude_vs_engagement.png")
    else:
        log.warning("Insufficient data for scatter plot.")

    # --- 3. Spike shape distribution bar chart ---------------------------
    log.info("Generating spike shape distribution ...")
    shape_counts = profiles_pd["spike_shape"].value_counts()
    shape_colors = {
        "sharp_spike": "#E74C3C",
        "sustained_plateau": "#3498DB",
        "double_peak": "#F39C12",
        "slow_burn": "#2ECC71",
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(
        shape_counts.index,
        shape_counts.values,
        color=[shape_colors.get(s, "#888") for s in shape_counts.index],
        edgecolor="white",
    )
    ax.set_xlabel("Spike Shape")
    ax.set_ylabel("Number of Anomaly Windows")
    ax.set_title("Distribution of Spike Shape Categories")
    for bar, val in zip(bars, shape_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                str(val), ha="center", fontweight="bold")
    save_fig(fig, "spike_shape_distribution.png")

    # ----- Summary stats -------------------------------------------------
    print("\n--- Stage 4 Summary ---")
    print(f"  Total spike profiles        : {len(profiles_pd):,}")
    print(f"  Spike shape distribution    :")
    for s, c in shape_counts.items():
        pct = 100 * c / len(profiles_pd)
        print(f"    {s:25s}: {c:5d} ({pct:.1f}%)")
    print(f"  Avg peak z-score            : {profiles_pd['peak_z_score'].mean():.2f}")
    print(f"  Avg post-spike engagement   : {profiles_pd['post_spike_avg_score'].mean():.2f}")
    print(f"  Elapsed time                : {time.time() - t0:.1f}s")

    spark.stop()
    log.info("Stage 4 complete.")


if __name__ == "__main__":
    main()
