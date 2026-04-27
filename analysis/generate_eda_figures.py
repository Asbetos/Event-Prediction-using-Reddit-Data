#!/usr/bin/env python3
"""Regenerate Stage 2-5 EDA figures locally from data/intermediate/.

Useful when rebuilding the website without re-running the Spark stages.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "intermediate"
FIG = ROOT / "website" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
GT = ROOT / "data" / "ground_truth" / "events.csv"

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CAT_COLORS = {
    "breaking_news": "#E74C3C",
    "controversy": "#F39C12",
    "product_launch": "#3498DB",
    "disaster": "#8E44AD",
    "meme_viral": "#2ECC71",
}
SHAPE_COLORS = {
    "sharp_spike": "#E74C3C",
    "sustained_plateau": "#3498DB",
    "double_peak": "#F39C12",
    "slow_burn": "#2ECC71",
}

plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name: str) -> None:
    out = FIG / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def figure_zscore_distribution() -> None:
    """Approximate the z-score histogram from anomaly_windows + hourly_counts."""
    aw = pd.read_parquet(DATA / "anomaly_windows.parquet")
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(0, 30, 80)
    ax.hist(aw["peak_z_score"].clip(upper=30), bins=bins, color="#3498DB", alpha=0.85)
    ax.axvline(3.0, color="#E74C3C", ls="--", lw=2, label="threshold (z=3)")
    ax.set_xlabel("Peak z-score per anomaly window")
    ax.set_ylabel("Number of windows")
    ax.set_title(f"Stage 2 — peak-z distribution ({len(aw):,} anomaly windows)")
    ax.set_yscale("log")
    ax.legend()
    save(fig, "zscore_distribution.png")


def figure_monthly_anomaly_counts() -> None:
    aw = pd.read_parquet(DATA / "anomaly_windows.parquet")
    aw["month"] = pd.to_datetime(aw["window_start"]).dt.strftime("%Y-%m")
    counts = aw.groupby("month").size().sort_index()
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(counts.index, counts.values, color="#2ECC71", edgecolor="white")
    ax.set_ylabel("Anomaly windows")
    ax.set_title("Stage 2 — anomaly windows per month")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save(fig, "monthly_anomaly_counts.png")


def figure_spike_shape_distribution() -> None:
    sp = pd.read_parquet(DATA / "spike_profiles.parquet")
    counts = sp["spike_shape"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[SHAPE_COLORS.get(s, "#888") for s in counts.index],
        edgecolor="white",
    )
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(counts) * 0.01, f"{v:,}", ha="center", fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("Anomaly windows (log)")
    ax.set_title(f"Stage 4 — spike-shape distribution ({len(sp):,} windows)")
    save(fig, "spike_shape_distribution.png")


def figure_spike_magnitude_vs_engagement() -> None:
    sp = pd.read_parquet(DATA / "spike_profiles.parquet")
    sp = sp.dropna(subset=["peak_z_score", "post_spike_avg_score"])
    sp = sp[(sp["post_spike_avg_score"] > 0) & (sp["peak_z_score"] < 200)]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.scatter(
        sp["peak_z_score"],
        sp["post_spike_avg_score"],
        alpha=0.25,
        s=12,
        c="#3498DB",
        edgecolors="none",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Peak z-score (log)")
    ax.set_ylabel("Post-spike avg score (log)")
    ax.set_title("Stage 4 — spike magnitude vs post-spike engagement")
    save(fig, "spike_magnitude_vs_engagement.png")


def figure_anomaly_heatmap() -> None:
    tp = pd.read_parquet(DATA / "temporal_patterns.parquet")
    grid = np.zeros((7, 24))
    for r in tp.itertuples():
        d, h = int(r.day_of_week), int(r.hour_of_day)
        if 0 <= d < 7 and 0 <= h < 24:
            grid[d, h] = float(r.anomaly_count)
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(7))
    ax.set_yticklabels(DOW)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_title("Stage 5 — anomaly density by day-of-week × hour")
    fig.colorbar(im, ax=ax, label="Anomaly count")
    save(fig, "anomaly_heatmap_dow_hour.png")


def figure_anomaly_polar() -> None:
    tp = pd.read_parquet(DATA / "temporal_patterns.parquet")
    by_hour = tp.groupby("hour_of_day")["anomaly_count"].sum().reindex(range(24), fill_value=0)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    vals = np.append(by_hour.values, by_hour.values[0])
    angs = np.append(angles, angles[0])
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.plot(angs, vals, color="#E74C3C", lw=2)
    ax.fill(angs, vals, alpha=0.25, color="#E74C3C")
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    ax.set_title("Stage 5 — anomaly frequency by hour-of-day (UTC)", pad=20)
    save(fig, "anomaly_polar_hour.png")


def figure_event_category_by_dow() -> None:
    if not GT.exists():
        return
    gt = pd.read_csv(GT)
    gt["date"] = pd.to_datetime(gt["date"])
    gt["dow"] = gt["date"].dt.dayofweek
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(7)
    for cat in sorted(gt["category"].unique()):
        sub = gt[gt["category"] == cat]
        counts = np.zeros(7)
        for d in range(7):
            counts[d] = (sub["dow"] == d).sum()
        ax.bar(range(7), counts, bottom=bottom, label=cat, color=CAT_COLORS.get(cat, "#888"))
        bottom += counts
    ax.set_xticks(range(7))
    ax.set_xticklabels(DOW)
    ax.set_ylabel("Ground-truth events")
    ax.set_title("Stage 5 — ground-truth events by day-of-week")
    ax.legend()
    save(fig, "event_category_by_dow.png")


def main() -> None:
    figure_zscore_distribution()
    figure_monthly_anomaly_counts()
    figure_spike_shape_distribution()
    figure_spike_magnitude_vs_engagement()
    figure_anomaly_heatmap()
    figure_anomaly_polar()
    figure_event_category_by_dow()


if __name__ == "__main__":
    main()
