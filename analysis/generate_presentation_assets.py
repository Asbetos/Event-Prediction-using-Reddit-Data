#!/usr/bin/env python3
"""Generate presentation-ready figures and metrics from local intermediate data.

This script reads only from data/intermediate/ and data/ground_truth/events.csv,
then writes a fresh presentation asset bundle under presentation_assets/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "intermediate"
GT = ROOT / "data" / "ground_truth" / "events.csv"
OUT_DIR = ROOT / "presentation_assets" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = ROOT / "presentation_assets" / "presentation_metrics.json"
IMG_DIR = ROOT / "presentation_assets" / "images"

PALETTE = {
    "ink": "#16324F",
    "slate": "#5B6B7A",
    "grid": "#D9E2EC",
    "paper": "#FFFFFF",
    "fog": "#F6F8FB",
    "orange": "#F45D48",
    "gold": "#F4B942",
    "blue": "#2B6CB0",
    "teal": "#2A9D8F",
    "purple": "#6C5B7B",
    "green": "#4C956C",
    "gray": "#C9D2DC",
    "dark_gray": "#8A94A6",
}

CATEGORY_COLORS = {
    "breaking_news": PALETTE["orange"],
    "controversy": PALETTE["gold"],
    "product_launch": PALETTE["blue"],
    "disaster": PALETTE["purple"],
    "meme_viral": PALETTE["teal"],
}

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CATEGORY_ORDER = [
    "breaking_news",
    "controversy",
    "product_launch",
    "disaster",
    "meme_viral",
]

plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 180,
        "font.family": "DejaVu Sans",
        "axes.facecolor": PALETTE["paper"],
        "figure.facecolor": PALETTE["paper"],
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "text.color": PALETTE["ink"],
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.45,
        "grid.linestyle": "-",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["paper"])
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def fmt_count(n: float | int) -> str:
    n = float(n)
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{int(n)}"


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["grid"])
    ax.spines["bottom"].set_color(PALETTE["grid"])
    ax.grid(axis=grid_axis)


def center_crop_to_ratio(image: Image.Image, target_ratio: float) -> Image.Image:
    width, height = image.size
    current_ratio = width / height
    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        return image.crop((left, 0, left + new_width, height))
    new_height = int(width / target_ratio)
    top = (height - new_height) // 2
    return image.crop((0, top, width, top + new_height))


def load_data() -> dict[str, pd.DataFrame]:
    frames = {
        "hourly": pd.read_parquet(DATA / "hourly_counts.parquet"),
        "anomaly": pd.read_parquet(DATA / "anomaly_windows.parquet"),
        "propagation": pd.read_parquet(DATA / "propagation_events.parquet"),
        "spike": pd.read_parquet(DATA / "spike_profiles.parquet"),
        "temporal": pd.read_parquet(DATA / "temporal_patterns.parquet"),
        "entities": pd.read_parquet(DATA / "entities.parquet"),
        "sentiment": pd.read_parquet(DATA / "sentiment.parquet"),
        "topics": pd.read_parquet(DATA / "topics.parquet"),
        "topic_details": pd.read_parquet(DATA / "topic_details.parquet"),
        "classifications": pd.read_parquet(DATA / "classifications.parquet"),
        "sustain": pd.read_parquet(DATA / "sustain_predictions.parquet"),
        "sustain_imp": pd.read_parquet(DATA / "sustain_feature_importance.parquet"),
        "forecast": pd.read_parquet(DATA / "forecast_results.parquet"),
        "forecast_imp": pd.read_parquet(DATA / "forecast_feature_importance.parquet"),
        "forecast_fold": pd.read_parquet(DATA / "forecast_fold_metrics.parquet"),
        "ground_truth": pd.read_csv(GT),
    }
    frames["ground_truth"]["date"] = pd.to_datetime(frames["ground_truth"]["date"])
    frames["anomaly"]["window_start"] = pd.to_datetime(frames["anomaly"]["window_start"])
    frames["anomaly"]["window_end"] = pd.to_datetime(frames["anomaly"]["window_end"])
    frames["hourly"]["hour_bucket"] = pd.to_datetime(frames["hourly"]["hour_bucket"])
    return frames


def compute_event_coverage(gt: pd.DataFrame, anomaly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in gt.itertuples(index=False):
        subreddits = str(r.relevant_subreddits).split("|")
        start = r.date - pd.Timedelta(hours=24)
        end = r.date + pd.Timedelta(hours=24)
        matched = anomaly[
            anomaly["subreddit"].isin(subreddits)
            & (anomaly["window_start"] <= end)
            & (anomaly["window_end"] >= start)
        ]
        rows.append(
            {
                "event_name": r.name,
                "category": r.category,
                "matched": int(not matched.empty),
                "matched_windows": int(len(matched)),
                "matched_subreddits": int(matched["subreddit"].nunique()),
                "max_peak_z": None if matched.empty else float(matched["peak_z_score"].max()),
            }
        )
    return pd.DataFrame(rows)


def build_metrics(frames: dict[str, pd.DataFrame]) -> dict:
    hourly = frames["hourly"]
    anomaly = frames["anomaly"]
    entities = frames["entities"]
    gt = frames["ground_truth"]
    coverage = compute_event_coverage(gt, anomaly)
    forecast = frames["forecast"]
    forecast_pos = forecast[forecast["label"] == 1].drop_duplicates(subset=["event_name"]).copy()

    metrics = {
        "dataset": {
            "comments_total": int(hourly.loc[hourly["data_type"] == "comments", "post_count"].sum()),
            "submissions_total": int(hourly.loc[hourly["data_type"] == "submissions", "post_count"].sum()),
            "total_posts": int(hourly["post_count"].sum()),
            "hourly_rows": int(len(hourly)),
            "subreddits": int(hourly["subreddit"].nunique()),
            "date_min": str(hourly["hour_bucket"].min()),
            "date_max": str(hourly["hour_bucket"].max()),
        },
        "ground_truth": {
            "n_events": int(len(gt)),
            "category_counts": gt["category"].value_counts().reindex(CATEGORY_ORDER).fillna(0).astype(int).to_dict(),
        },
        "stage2": {
            "anomaly_windows": int(len(anomaly)),
            "subreddits_with_anomaly": int(anomaly["subreddit"].nunique()),
            "median_peak_z": float(anomaly["peak_z_score"].median()),
            "p95_peak_z": float(anomaly["peak_z_score"].quantile(0.95)),
        },
        "coverage": {
            "matched_events": int(coverage["matched"].sum()),
            "total_events": int(len(coverage)),
            "by_category": coverage.groupby("category")["matched"].agg(["sum", "count"]).reindex(CATEGORY_ORDER).fillna(0).astype(int).to_dict(orient="index"),
        },
        "stage6": {
            "entity_rows": int(len(entities)),
            "unique_entities": int(entities["entity_text"].nunique()),
            "windows_with_entities": int(entities["window_id"].nunique()),
        },
        "stage7": {
            "rows": int(len(frames["sentiment"])),
        },
        "stage8": {
            "rows": int(len(frames["topics"])),
        },
        "stage9": {
            "predictions": int(len(frames["classifications"])),
        },
        "stage10": {
            "predictions": int(len(frames["sustain"])),
            "auc": float(frames["sustain"]["model_roc_auc"].iloc[0]),
            "f1": float(frames["sustain"]["model_f1"].iloc[0]),
        },
        "stage11": {
            "predictions": int(len(forecast)),
            "positives": int(forecast["label"].sum()),
            "negatives": int((forecast["label"] == 0).sum()),
            "auc": float(forecast["model_roc_auc"].iloc[0]),
            "ap": float(forecast["model_avg_precision"].iloc[0]),
            "precision": float(forecast["model_precision"].iloc[0]),
            "recall": float(forecast["model_recall"].iloc[0]),
            "f1": float(forecast["model_f1"].iloc[0]),
            "event_probability_by_category": forecast_pos.groupby("event_category")["event_probability"].mean().reindex(CATEGORY_ORDER).dropna().to_dict(),
        },
    }
    return metrics


def figure_pipeline_scale(metrics: dict) -> None:
    dataset = metrics["dataset"]
    values = [
        dataset["total_posts"],
        dataset["hourly_rows"],
        metrics["stage2"]["anomaly_windows"],
        metrics["stage6"]["windows_with_entities"],
        metrics["stage11"]["predictions"],
        metrics["ground_truth"]["n_events"],
    ]
    labels = [
        "Raw Reddit posts",
        "Hourly subreddit rows",
        "Anomaly windows",
        "NER-covered windows",
        "Forecast examples",
        "Curated ground-truth events",
    ]
    colors = [
        PALETTE["ink"],
        PALETTE["blue"],
        PALETTE["gold"],
        PALETTE["teal"],
        PALETTE["purple"],
        PALETTE["orange"],
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="white", height=0.7)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count (log scale)")
    ax.set_title("From 1.27B Reddit posts to a compact forecasting dataset")
    style_axis(ax, grid_axis="x")
    for bar, value in zip(bars, values):
        ax.text(value * 1.08, bar.get_y() + bar.get_height() / 2, fmt_count(value), va="center", fontsize=10)
    fig.text(
        0.5,
        0.01,
        "Each downstream stage works on a much smaller, more structured slice of the data.",
        ha="center",
        fontsize=10,
        color=PALETTE["slate"],
    )
    save(fig, "pipeline_scale_funnel.png")


def figure_ground_truth_overview(gt: pd.DataFrame) -> None:
    counts = gt["category"].value_counts().reindex(CATEGORY_ORDER).fillna(0).astype(int)
    ymap = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [0.9, 1.4]})

    ax = axes[0]
    bars = ax.barh(
        counts.index,
        counts.values,
        color=[CATEGORY_COLORS[c] for c in counts.index],
        edgecolor="white",
        height=0.7,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Event count")
    ax.set_title("Ground-truth event mix")
    style_axis(ax, grid_axis="x")
    for bar, value in zip(bars, counts.values):
        ax.text(value + 0.15, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center", fontsize=10)

    ax = axes[1]
    for cat in CATEGORY_ORDER:
        sub = gt[gt["category"] == cat]
        ax.scatter(
            sub["date"],
            np.full(len(sub), ymap[cat]),
            s=110,
            color=CATEGORY_COLORS[cat],
            edgecolors="white",
            linewidths=0.8,
            alpha=0.95,
            label=cat.replace("_", " "),
        )
    ax.set_yticks(range(len(CATEGORY_ORDER)))
    ax.set_yticklabels([c.replace("_", " ") for c in CATEGORY_ORDER])
    ax.set_xlabel("Event date")
    ax.set_title("35 curated events across June 2023 to July 2024")
    style_axis(ax, grid_axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    save(fig, "ground_truth_overview.png")


def figure_event_story_collage() -> None:
    tiles = [
        ("reddit.png", "Reddit blackout\ncontroversy"),
        ("titan.jpg", "Titan implosion\ndisaster"),
        ("sam_altman.jpg", "OpenAI leadership drama\ncontroversy"),
        ("eclipse.jpg", "North America eclipse\nbreaking news"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
    for ax, (filename, label) in zip(axes.flat, tiles):
        path = IMG_DIR / filename
        image = Image.open(path).convert("RGB")
        image = center_crop_to_ratio(image, 1.35)
        ax.imshow(np.asarray(image))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            0.03,
            0.08,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#12263A", "edgecolor": "none", "alpha": 0.88},
        )
    fig.suptitle("Representative events from the curated catalog", fontsize=16, fontweight="bold", y=0.98)
    save(fig, "event_story_collage.png")


def figure_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, face: str, text_size: int = 12) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.18",
            linewidth=1.2,
            edgecolor=PALETTE["grid"],
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=text_size, fontweight="bold")

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2,
            color=PALETTE["slate"],
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(patch)

    ax.text(0.4, 9.45, "Project architecture", fontsize=20, fontweight="bold", color=PALETTE["ink"])
    ax.text(0.4, 8.95, "Two machines, 11 stages, one staged-data contract", fontsize=11, color=PALETTE["slate"])

    box(0.5, 5.4, 2.6, 1.7, "Raw Reddit\n478 GB parquet", "#EEF4FB", 14)
    box(3.7, 5.4, 2.5, 1.7, "Stage 1\nGPU aggregation", "#DDEBFB", 13)

    box(6.9, 6.8, 2.6, 1.45, "Stages 2-5\nSpark EDA", "#FFF0D4", 13)
    box(6.9, 4.55, 2.6, 1.45, "Stages 6-8\nGPU NLP", "#DDF3EF", 13)

    box(10.3, 6.8, 2.7, 1.45, "Stages 9-10\nclassification + sustain", "#F4E6F8", 12)
    box(10.3, 4.55, 2.7, 1.45, "Stage 11\n48h forecasting", "#E7F0FB", 13)
    box(10.3, 2.3, 2.7, 1.45, "Ground truth\n35 curated events", "#F9E2DE", 13)

    box(13.7, 5.4, 1.9, 1.7, "Figures\n+ deck", "#EEF4FB", 14)

    arrow(3.1, 6.25, 3.7, 6.25)
    arrow(6.2, 6.25, 6.9, 7.5)
    arrow(6.2, 6.25, 6.9, 5.25)
    arrow(9.5, 7.5, 10.3, 7.5)
    arrow(9.5, 5.25, 10.3, 5.25)
    arrow(11.65, 3.75, 11.65, 4.55)
    arrow(13.0, 7.5, 13.7, 6.25)
    arrow(13.0, 5.25, 13.7, 6.0)

    box(5.9, 8.7, 3.1, 0.7, "EC2 t3.large · PySpark", "#F9FBFD", 11)
    box(5.9, 3.55, 3.1, 0.7, "RunPod A100 · RAPIDS + transformers", "#F9FBFD", 11)

    save(fig, "pipeline_diagram.png")


def figure_detection_overview(anomaly: pd.DataFrame, gt: pd.DataFrame) -> None:
    coverage = compute_event_coverage(gt, anomaly)
    monthly = anomaly.assign(month=anomaly["window_start"].dt.strftime("%Y-%m")).groupby("month").size().sort_index()
    cov = (
        coverage.groupby("category")["matched"]
        .agg([("matched", "sum"), ("total", "count")])
        .reindex(CATEGORY_ORDER)
        .fillna(0)
        .astype(int)
    )
    cov["unmatched"] = cov["total"] - cov["matched"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.0, 1.25, 1.0]})

    ax = axes[0]
    bins = np.linspace(0, 30, 70)
    ax.hist(anomaly["peak_z_score"].clip(upper=30), bins=bins, color=PALETTE["blue"], edgecolor="white", alpha=0.95)
    ax.axvline(3.0, color=PALETTE["orange"], linestyle="--", linewidth=2, label="z = 3 threshold")
    ax.set_yscale("log")
    ax.set_xlabel("Peak z-score per anomaly window")
    ax.set_ylabel("Window count")
    ax.set_title("Detected anomalies are concentrated near threshold\nwith a long, meaningful tail")
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    month_labels = list(monthly.index)
    month_colors = [PALETTE["blue"]] * len(month_labels)
    for i, m in enumerate(month_labels):
        if m in {"2023-06", "2023-10"}:
            month_colors[i] = PALETTE["orange"]
    ax.bar(month_labels, monthly.values, color=month_colors, edgecolor="white")
    ax.set_title("Monthly anomaly volume tracks real news cycles")
    ax.set_ylabel("Anomaly windows")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    style_axis(ax, grid_axis="y")

    ax = axes[2]
    ypos = np.arange(len(cov.index))
    ax.barh(ypos, cov["matched"], color=PALETTE["teal"], edgecolor="white", label="matched")
    ax.barh(ypos, cov["unmatched"], left=cov["matched"], color=PALETTE["gray"], edgecolor="white", label="unmatched")
    ax.set_yticks(ypos)
    ax.set_yticklabels([c.replace("_", " ") for c in cov.index])
    ax.invert_yaxis()
    ax.set_xlabel("Ground-truth events")
    ax.set_title(f"Conservative alignment check: {int(coverage['matched'].sum())} / {len(coverage)} events\nshow a subreddit anomaly within +/-24 hours")
    style_axis(ax, grid_axis="x")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "detection_overview.png")


def figure_temporal_overview(temporal: pd.DataFrame, gt: pd.DataFrame) -> None:
    grid = np.zeros((7, 24))
    for row in temporal.itertuples(index=False):
        d = int(row.day_of_week)
        h = int(row.hour_of_day)
        grid[d, h] = float(row.anomaly_count)

    gt = gt.copy()
    gt["dow"] = gt["date"].dt.dayofweek

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list("story_heat", [PALETTE["fog"], PALETTE["gold"], PALETTE["orange"]])
    im = ax.imshow(grid, aspect="auto", cmap=cmap)
    ax.set_yticks(range(7))
    ax.set_yticklabels(DOW)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_title("Reddit anomalies peak midweek and around US/EU overlap hours")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Anomaly count")

    ax = axes[1]
    bottom = np.zeros(7)
    for cat in CATEGORY_ORDER:
        sub = gt[gt["category"] == cat]
        counts = np.array([(sub["dow"] == d).sum() for d in range(7)])
        ax.bar(range(7), counts, bottom=bottom, color=CATEGORY_COLORS[cat], edgecolor="white", label=cat.replace("_", " "))
        bottom += counts
    ax.set_xticks(range(7))
    ax.set_xticklabels(DOW)
    ax.set_ylabel("Ground-truth events")
    ax.set_title("Breaking news and controversies skew midweek")
    style_axis(ax, grid_axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    save(fig, "temporal_overview.png")


def figure_entities_overview(entities: pd.DataFrame) -> None:
    label_counts = entities["entity_label"].value_counts().reindex(["PERSON", "ORG", "GPE"]).fillna(0)

    selected = ["Israel", "Hamas", "Reddit", "Gaza", "Trump", "NFL", "Palestine", "IDF", "Microsoft"]
    tmp = entities.copy()
    tmp["entity_key"] = tmp["entity_text"].str.lower()
    rows = []
    for label in selected:
        key = label.lower()
        sub = tmp[tmp["entity_key"] == key]
        if sub.empty:
            continue
        by_label = sub.groupby("entity_label")["count"].sum()
        dominant = by_label.idxmax()
        rows.append({"entity": label, "count": int(sub["count"].sum()), "dominant_label": dominant})
    selected_df = pd.DataFrame(rows).sort_values("count", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [0.9, 1.35]})

    ax = axes[0]
    wedges, _ = ax.pie(
        label_counts.values,
        colors=[PALETTE["blue"], PALETTE["orange"], PALETTE["teal"]],
        startangle=90,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
    )
    ax.set_title("NER output mix across 63,614 extracted entities")
    ax.legend(
        wedges,
        [f"{k} ({int(v):,})" for k, v in label_counts.items()],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        ncol=1,
    )

    ax = axes[1]
    bar_colors = [
        PALETTE["blue"] if x == "PERSON" else PALETTE["orange"] if x == "ORG" else PALETTE["teal"]
        for x in selected_df["dominant_label"]
    ]
    ax.barh(selected_df["entity"], selected_df["count"], color=bar_colors, edgecolor="white")
    ax.set_xlabel("Mentions across anomaly windows")
    ax.set_title("Illustrative entities show real-world stories, not just volume spikes")
    style_axis(ax, grid_axis="x")
    save(fig, "entities_overview.png")


def figure_ml_pre_forecast(classifications: pd.DataFrame, sustain: pd.DataFrame, sustain_imp: pd.DataFrame) -> None:
    counts = classifications["predicted_category"].value_counts().reindex(CATEGORY_ORDER).fillna(0)
    cm = confusion_matrix(sustain["sustain_label"], sustain["predicted_label"], labels=["decayed", "sustained"])
    imp = sustain_imp.sort_values("importance", ascending=True).tail(5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw={"width_ratios": [0.9, 0.85, 1.0]})

    ax = axes[0]
    ax.bar(counts.index, counts.values, color=[CATEGORY_COLORS[c] for c in counts.index], edgecolor="white")
    for i, v in enumerate(counts.values):
        if v > 0:
            ax.text(i, v + 6, f"{int(v):,}", ha="center", fontsize=10)
    ax.set_ylabel("Predictions")
    ax.set_title("Stage 9: multi-class classification\ncollapses to the majority class")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    style_axis(ax, grid_axis="y")

    ax = axes[1]
    cmap = LinearSegmentedColormap.from_list("sustain", [PALETTE["fog"], PALETTE["gold"], PALETTE["orange"]])
    ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["decayed", "sustained"])
    ax.set_yticklabels(["decayed", "sustained"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        "Stage 10: ranking signal exists,\nbut thresholded positives fail\n"
        f"AUC={sustain['model_roc_auc'].iloc[0]:.3f} | F1={sustain['model_f1'].iloc[0]:.1f}"
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontweight="bold", fontsize=13)
    ax.grid(False)

    ax = axes[2]
    ax.barh(imp["feature"], imp["importance"], color=PALETTE["teal"], edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title("Stage 10: top early-spike features")
    style_axis(ax, grid_axis="x")
    save(fig, "ml_pre_forecast_overview.png")


def figure_forecast_overview(forecast: pd.DataFrame, forecast_imp: pd.DataFrame, forecast_fold: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5))

    ax = axes[0, 0]
    metrics = ["auc", "precision", "recall", "f1"]
    metric_colors = {
        "auc": PALETTE["blue"],
        "precision": PALETTE["teal"],
        "recall": PALETTE["gold"],
        "f1": PALETTE["purple"],
    }
    width = 0.18
    x = np.arange(len(forecast_fold))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width - 1.5 * width, forecast_fold[metric], width, label=metric.upper(), color=metric_colors[metric], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {int(f)}" for f in forecast_fold["fold"]])
    ax.set_ylim(0, 1.05)
    ax.set_title("Stable held-out performance across 5 folds")
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, ncol=2, loc="lower right")

    ax = axes[0, 1]
    cm = confusion_matrix(forecast["label"], forecast["predicted_event"], labels=[0, 1])
    cmap = LinearSegmentedColormap.from_list("forecast", [PALETTE["fog"], PALETTE["teal"], PALETTE["ink"]])
    ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["no event", "event"])
    ax.set_yticklabels(["no event", "event"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"Forecast confusion matrix\nAUC={forecast['model_roc_auc'].iloc[0]:.3f} | AP={forecast['model_avg_precision'].iloc[0]:.3f}"
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontweight="bold", fontsize=14, color=PALETTE["ink"])
    ax.grid(False)

    ax = axes[1, 0]
    fpr, tpr, _ = roc_curve(forecast["label"], forecast["event_probability"])
    ax.plot(fpr, tpr, color=PALETTE["orange"], linewidth=2.5, label=f"AUC = {forecast['model_roc_auc'].iloc[0]:.3f}")
    ax.fill_between(fpr, tpr, color=PALETTE["orange"], alpha=0.15)
    ax.plot([0, 1], [0, 1], color=PALETTE["dark_gray"], linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Forecast ROC curve")
    style_axis(ax, grid_axis="both")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 1]
    imp = forecast_imp.sort_values("importance", ascending=True).tail(7)
    ax.barh(imp["feature"], imp["importance"], color=PALETTE["blue"], edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title("Top forecast features")
    style_axis(ax, grid_axis="x")

    fig.suptitle(
        "Stage 11 is the strongest result: 48-hour precursor forecasting",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    save(fig, "forecast_overview.png")


def figure_forecast_event_examples(forecast: pd.DataFrame) -> None:
    pos = forecast[forecast["label"] == 1].drop_duplicates(subset=["event_name"]).copy()
    pos = pos.sort_values("event_probability", ascending=False)
    top = pos.head(8).sort_values("event_probability", ascending=True)
    bottom = pos.tail(8).sort_values("event_probability", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    for ax, subset, title in [
        (axes[0], top, "Highest-scored real events"),
        (axes[1], bottom, "Lowest-scored real events"),
    ]:
        colors = [PALETTE["teal"] if p >= 0.5 else PALETTE["orange"] for p in subset["event_probability"]]
        ax.barh(subset["event_name"].str.slice(0, 48), subset["event_probability"], color=colors, edgecolor="white")
        ax.axvline(0.5, color=PALETTE["dark_gray"], linestyle="--", linewidth=1.6)
        ax.set_title(title)
        ax.set_xlabel("Held-out event probability")
        style_axis(ax, grid_axis="x")
    save(fig, "forecast_event_examples.png")


def main() -> None:
    frames = load_data()
    metrics = build_metrics(frames)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {METRICS_PATH.relative_to(ROOT)}")

    figure_pipeline_scale(metrics)
    figure_ground_truth_overview(frames["ground_truth"])
    figure_event_story_collage()
    figure_pipeline_diagram()
    figure_detection_overview(frames["anomaly"], frames["ground_truth"])
    figure_temporal_overview(frames["temporal"], frames["ground_truth"])
    figure_entities_overview(frames["entities"])
    figure_ml_pre_forecast(frames["classifications"], frames["sustain"], frames["sustain_imp"])
    figure_forecast_overview(frames["forecast"], frames["forecast_imp"], frames["forecast_fold"])
    figure_forecast_event_examples(frames["forecast"])


if __name__ == "__main__":
    main()
