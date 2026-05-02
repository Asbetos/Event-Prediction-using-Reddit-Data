#!/usr/bin/env python3
"""Build all report figures referenced in the final report from Code/data/intermediate/."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, roc_curve

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DATA = ROOT / "data" / "intermediate"
FIG = REPO / "Final-Project-Report" / "figures" / "report"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.figsize": (12, 6),
        "figure.dpi": 150,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def save(fig, name: str) -> None:
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


# =============================================================================
# Stage 6: Named Entity Recognition
# =============================================================================
def figure_entities() -> None:
    print("[NER] building entity figures...")
    ents = pd.read_parquet(DATA / "entities.parquet")

    # 1. Top 15 entities per type
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    palette = {"PERSON": "#3498DB", "ORG": "#E74C3C", "GPE": "#27AE60"}
    for ax, label in zip(axes, ["PERSON", "ORG", "GPE"]):
        sub = (
            ents[ents["entity_label"] == label]
            .groupby("entity_text")["count"]
            .sum()
            .sort_values(ascending=True)
            .tail(15)
        )
        ax.barh(sub.index, sub.values, color=palette[label], edgecolor="white")
        ax.set_title(f"Top 15 {label} entities (Stage 6)")
        ax.set_xlabel("Total mentions across anomaly windows")
    save(fig, "entities_top_by_type.png")

    # 2. Label distribution
    label_counts = ents["entity_label"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        label_counts.index,
        label_counts.values,
        color=[palette.get(l, "#888") for l in label_counts.index],
        edgecolor="white",
    )
    for i, v in enumerate(label_counts.values):
        ax.text(i, v + max(label_counts) * 0.01, f"{v:,}", ha="center", fontweight="bold")
    ax.set_ylabel("Entity rows")
    ax.set_title(f"Stage 6 — entity label distribution ({len(ents):,} rows)")
    save(fig, "entities_label_distribution.png")


# =============================================================================
# Stage 7: Sentiment (very partial — only 2 windows completed)
# =============================================================================
def figure_sentiment() -> None:
    print("[SENT] building sentiment figure...")
    sent = pd.read_parquet(DATA / "sentiment.parquet")

    rows = sent.head(2)  # we have 2 rows
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rows))
    width = 0.25
    ax.bar(
        x - width,
        rows["anomaly_prop_positive"],
        width,
        label="positive",
        color="#27AE60",
    )
    ax.bar(
        x,
        rows["anomaly_prop_neutral"],
        width,
        label="neutral",
        color="#95A5A6",
    )
    ax.bar(
        x + width,
        rows["anomaly_prop_negative"],
        width,
        label="negative",
        color="#E74C3C",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"r/{r.subreddit}\n{str(r.start_time)[:10]}" for r in rows.itertuples()],
        fontsize=10,
    )
    ax.set_ylabel("Class proportion")
    ax.set_title("Stage 7 — sentiment proportions (anomaly window, partial run)")
    ax.legend()
    save(fig, "sentiment_class_proportions.png")


# =============================================================================
# Stage 8: Topic modeling
# =============================================================================
def figure_topics() -> None:
    print("[TOPICS] building topic figure...")
    topics = pd.read_parquet(DATA / "topics.parquet")
    details = pd.read_parquet(DATA / "topic_details.parquet")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    counts = topics["dominant_topic"].value_counts().sort_index()
    colors = ["#bdc3c7" if t == -1 else "#3498DB" for t in counts.index]
    ax1.bar(
        [str(t) for t in counts.index],
        counts.values,
        color=colors,
        edgecolor="white",
    )
    ax1.set_xlabel("Dominant topic id  (−1 = noise / unclustered)")
    ax1.set_ylabel("Anomaly windows")
    ax1.set_title(f"Stage 8 — topic assignment ({len(topics)} windows)")
    for i, v in enumerate(counts.values):
        ax1.text(i, v + 0.3, str(int(v)), ha="center", fontweight="bold")

    ax2.axis("off")
    ax2.set_title("Stage 8 — representative words per topic")
    rows = []
    for r in details.itertuples():
        rows.append([f"topic {r.topic_id}", f"{int(r.count):,}", str(r.representative_words)[:120]])
    if rows:
        table = ax2.table(
            cellText=rows,
            colLabels=["topic", "doc count", "representative words"],
            cellLoc="left",
            colLoc="left",
            loc="center",
            colWidths=[0.12, 0.12, 0.76],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.7)
    save(fig, "topics_distribution.png")


# =============================================================================
# Stage 9: Classification (degenerate — single class predicted)
# =============================================================================
def figure_classification() -> None:
    print("[CLF] building classification figure...")
    clf = pd.read_parquet(DATA / "classifications.parquet")
    counts = clf["predicted_category"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color="#9B59B6", edgecolor="white")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, f"{v:,}", ha="center", fontweight="bold")
    ax.set_ylabel("Predictions")
    ax.set_title(
        f"Stage 9 — predicted event category ({len(clf):,} windows)"
    )
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    save(fig, "classification_predicted_categories.png")


# =============================================================================
# Stage 10: Sustain/decay prediction
# =============================================================================
def figure_sustain() -> None:
    print("[SUSTAIN] building sustain figures...")
    sus = pd.read_parquet(DATA / "sustain_predictions.parquet")
    imp = pd.read_parquet(DATA / "sustain_feature_importance.parquet").sort_values(
        "importance", ascending=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # confusion matrix
    cm = confusion_matrix(sus["sustain_label"], sus["predicted_label"], labels=["decayed", "sustained"])
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list("blues", ["#ecf6ff", "#3498DB", "#1f5d8a"])
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["decayed", "sustained"])
    ax.set_yticklabels(["decayed", "sustained"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"Stage 10 — confusion matrix (n={len(sus):,}, AUC={sus['model_roc_auc'].iloc[0]:.3f})"
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontweight="bold", fontsize=14)

    # feature importance
    ax = axes[1]
    ax.barh(imp["feature"], imp["importance"], color="#16A085", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title("Stage 10 — feature importance (cuML RF)")
    save(fig, "sustain_confusion_and_importance.png")


# =============================================================================
# Stage 11: Forecasting — the strongest result
# =============================================================================
def figure_forecast() -> None:
    print("[FORECAST] building forecast figures...")
    fc = pd.read_parquet(DATA / "forecast_results.parquet")
    fc_imp = pd.read_parquet(DATA / "forecast_feature_importance.parquet").sort_values(
        "importance", ascending=True
    )
    fc_fold = pd.read_parquet(DATA / "forecast_fold_metrics.parquet")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # per-fold metrics
    ax = axes[0, 0]
    metrics = ["auc", "precision", "recall", "f1"]
    width = 0.18
    x = np.arange(len(fc_fold))
    colors = {"auc": "#3498DB", "precision": "#27AE60", "recall": "#F39C12", "f1": "#9B59B6"}
    for i, m in enumerate(metrics):
        ax.bar(x + i * width - 1.5 * width, fc_fold[m], width, label=m, color=colors[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {f}" for f in fc_fold["fold"]])
    ax.set_ylim(0, 1.05)
    ax.set_title("Stage 11 — 5-fold cross-validation metrics")
    ax.legend(loc="lower right")

    # confusion matrix
    cm = confusion_matrix(fc["label"], fc["predicted_event"], labels=[0, 1])
    ax = axes[0, 1]
    cmap = LinearSegmentedColormap.from_list("greens", ["#eafaf1", "#27AE60", "#145a32"])
    ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["no event", "event"])
    ax.set_yticklabels(["no event", "event"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    auc = fc["model_roc_auc"].iloc[0]
    ap = fc["model_avg_precision"].iloc[0]
    ax.set_title(
        f"Stage 11 — confusion matrix (n={len(fc):,}, AUC={auc:.3f}, AP={ap:.3f})"
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontweight="bold", fontsize=14)

    # ROC curve
    ax = axes[1, 0]
    fpr, tpr, _ = roc_curve(fc["label"], fc["event_probability"])
    ax.plot(fpr, tpr, color="#E74C3C", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#888", linestyle="--", alpha=0.6)
    ax.fill_between(fpr, tpr, alpha=0.2, color="#E74C3C")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Stage 11 — ROC curve")
    ax.legend(loc="lower right")

    # feature importance
    ax = axes[1, 1]
    ax.barh(fc_imp["feature"], fc_imp["importance"], color="#2980B9", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title("Stage 11 — feature importance")

    save(fig, "forecast_summary.png")


# =============================================================================
# Stage 11: events × predicted-probability scatter for narrative
# =============================================================================
def figure_forecast_events() -> None:
    print("[FORECAST] building event-level scatter...")
    fc = pd.read_parquet(DATA / "forecast_results.parquet")
    pos = fc[fc["label"] == 1].copy()
    if pos.empty:
        return
    pos = pos.sort_values("event_probability", ascending=True).head(33)

    fig, ax = plt.subplots(figsize=(10, 12))
    bars = ax.barh(
        pos["event_name"].fillna("(unknown)").str.slice(0, 50),
        pos["event_probability"],
        color=["#27AE60" if p >= 0.5 else "#E74C3C" for p in pos["event_probability"]],
        edgecolor="white",
    )
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Predicted event probability (held-out fold)")
    ax.set_title(
        f"Stage 11 — per-event hold-out predictions (n={len(pos)} ground-truth events)"
    )
    ax.set_xlim(0, 1)
    save(fig, "forecast_events_per_label.png")


# =============================================================================
# Cross-stage summary panel
# =============================================================================
def figure_pipeline_overview() -> None:
    print("[OVERVIEW] building stage-volume figure...")
    rows = [
        ("Stage 1: aggregation", 1_274_668_346, "#3498DB"),
        ("Stage 2: anomalies", 31938, "#E74C3C"),
        ("Stage 3: clusters", 1, "#9B59B6"),
        ("Stage 4: spike profiles", 31938, "#F39C12"),
        ("Stage 5: temporal cells", 168, "#16A085"),
        ("Stage 6: entity rows", 63614, "#2ECC71"),
        ("Stage 7: sentiment rows", 2, "#34495E"),
        ("Stage 8: topic rows", 50, "#E67E22"),
        ("Stage 9: classifications", 500, "#1ABC9C"),
        ("Stage 10: sustain preds", 500, "#7D3C98"),
        ("Stage 11: forecast preds", 138, "#C0392B"),
    ]
    labels = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(labels, counts, color=colors, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Output rows (log scale)")
    ax.set_title("Pipeline output volume per stage")
    for i, v in enumerate(counts):
        ax.text(v * 1.1, i, f"{v:,}", va="center", fontsize=9)
    save(fig, "pipeline_output_volume.png")


def main() -> None:
    print(f"DATA={DATA}\nFIG ={FIG}")
    figure_pipeline_overview()
    figure_entities()
    figure_sentiment()
    figure_topics()
    figure_classification()
    figure_sustain()
    figure_forecast()
    figure_forecast_events()
    print("Done.")


if __name__ == "__main__":
    main()
