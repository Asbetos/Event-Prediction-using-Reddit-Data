#!/usr/bin/env python3
"""
Stage 3 -- Cross-Subreddit Propagation Analysis (Q2)
=====================================================
Identifies co-occurring anomaly windows across subreddits, clusters them
into event-level groups using NetworkX connected components, and classifies
each cluster's propagation type (niche-to-mainstream, simultaneous, top-down).

Outputs:
    propagation_events.parquet (local intermediate)
    Figures: propagation_type_distribution.png,
             propagation_scatter.png,
             propagation_network_top10.png
"""

import os, sys, logging, time
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config.spark_config import create_spark_session
from config.settings import TOP_N_SUBREDDITS
from utils.spark_utils import read_intermediate, write_intermediate
from utils.viz_utils import save_fig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("stage3")

CO_OCCURRENCE_WINDOW_HOURS = 48  # max gap to consider two anomalies related


def main():
    t0 = time.time()
    spark = create_spark_session(app_name="Stage3_Propagation")
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

    log.info("Anomaly windows: %s rows", f"{anomaly_windows.count():,}")

    # Read subreddit stats for size-based classification
    try:
        sub_stats = read_intermediate(spark, "subreddit_stats.parquet")
    except Exception:
        log.warning("subreddit_stats.parquet not found; deriving from hourly_counts.")
        sub_stats = (
            hourly.groupBy("subreddit")
            .agg(F.sum("post_count").alias("total_posts"))
        )

    # Ensure total_posts is available
    if "total_posts" not in sub_stats.columns:
        sub_stats = sub_stats.withColumnRenamed(
            [c for c in sub_stats.columns if "post" in c.lower()][0],
            "total_posts",
        )

    sub_stats = sub_stats.select("subreddit", "total_posts")

    # ----- 2. Self-join: find co-occurring anomalies ---------------------
    aw = anomaly_windows.select(
        F.col("subreddit").alias("sub_a"),
        F.col("window_id").alias("wid_a"),
        F.col("window_start").alias("start_a"),
        F.col("window_end").alias("end_a"),
        F.col("peak_z_score").alias("peak_a"),
    )

    bw = anomaly_windows.select(
        F.col("subreddit").alias("sub_b"),
        F.col("window_id").alias("wid_b"),
        F.col("window_start").alias("start_b"),
        F.col("window_end").alias("end_b"),
        F.col("peak_z_score").alias("peak_b"),
    )

    gap_seconds = CO_OCCURRENCE_WINDOW_HOURS * 3600

    # Overlapping or within 48h of each other
    co_occur = (
        aw.join(bw, on=(aw["sub_a"] < bw["sub_b"]))
        .filter(
            # windows overlap or are within gap
            (F.unix_timestamp(F.col("start_a").cast("timestamp")) - F.unix_timestamp(F.col("end_b").cast("timestamp")) <= gap_seconds)
            & (F.unix_timestamp(F.col("start_b").cast("timestamp")) - F.unix_timestamp(F.col("end_a").cast("timestamp")) <= gap_seconds)
        )
        .select("sub_a", "wid_a", "start_a", "sub_b", "wid_b", "start_b")
    )

    co_occur_pd = co_occur.toPandas()
    log.info("Co-occurring pairs: %s", f"{len(co_occur_pd):,}")

    if co_occur_pd.empty:
        log.warning("No co-occurring anomalies found. Writing empty output.")
        empty_schema = (
            "event_cluster_id string, subreddit_sequence string, "
            "propagation_type string, num_subreddits int, "
            "total_duration_hours double, first_detection_time timestamp"
        )
        empty_df = spark.createDataFrame([], schema=empty_schema)
        write_intermediate(empty_df, "propagation_events.parquet")
        spark.stop()
        return

    # ----- 3. Build graph & connected components -------------------------
    import networkx as nx

    G = nx.Graph()
    for _, row in co_occur_pd.iterrows():
        G.add_edge(row["wid_a"], row["wid_b"])

    components = list(nx.connected_components(G))
    log.info("Event clusters (connected components): %d", len(components))

    # Map window_id -> component id
    wid_to_cluster = {}
    for cid, comp in enumerate(components):
        for wid in comp:
            wid_to_cluster[wid] = cid

    # ----- 4. Build cluster metadata in Spark ----------------------------
    # Broadcast the mapping
    import pandas as pd

    cluster_map_pd = pd.DataFrame(
        list(wid_to_cluster.items()), columns=["window_id", "event_cluster_id"]
    )
    cluster_map_sdf = spark.createDataFrame(cluster_map_pd)

    aw_full = anomaly_windows.join(cluster_map_sdf, on="window_id", how="inner")

    # Join subreddit size
    aw_full = aw_full.join(sub_stats, on="subreddit", how="left")

    # Compute percentiles for propagation classification
    stats_row = sub_stats.agg(
        F.expr("percentile_approx(total_posts, 0.5)").alias("median_posts"),
        F.expr("percentile_approx(total_posts, 0.75)").alias("p75_posts"),
    ).collect()[0]
    median_posts = stats_row["median_posts"]
    p75_posts = stats_row["p75_posts"]
    log.info("Subreddit total_posts median=%.0f, p75=%.0f", median_posts, p75_posts)

    # Per-cluster aggregation
    w_cluster = Window.partitionBy("event_cluster_id")

    aw_full = aw_full.withColumn(
        "rank_in_cluster",
        F.row_number().over(
            Window.partitionBy("event_cluster_id").orderBy("window_start")
        ),
    )

    cluster_agg = (
        aw_full.groupBy("event_cluster_id")
        .agg(
            F.count("*").alias("num_subreddits"),
            F.min("window_start").alias("first_detection_time"),
            F.max("window_end").alias("last_end_time"),
            F.collect_list(
                F.struct("window_start", "subreddit", "total_posts")
            ).alias("members"),
        )
        .withColumn(
            "total_duration_hours",
            (F.unix_timestamp(F.col("last_end_time").cast("timestamp"))
             - F.unix_timestamp(F.col("first_detection_time").cast("timestamp"))) / 3600,
        )
    )

    cluster_pd = cluster_agg.toPandas()

    # ----- 5. Classify propagation type & build sequence -----------------
    records = []
    for _, row in cluster_pd.iterrows():
        members = sorted(row["members"], key=lambda m: m["window_start"])
        subreddit_seq = [m["subreddit"] for m in members]
        first_mover_posts = members[0]["total_posts"] or 0
        first_start = members[0]["window_start"]

        # Check simultaneity: >50% within 2h of first start
        within_2h = sum(
            1 for m in members
            if abs((m["window_start"] - first_start).total_seconds()) <= 7200
        )
        simultaneous_frac = within_2h / len(members) if members else 0

        if simultaneous_frac > 0.5:
            ptype = "simultaneous"
        elif first_mover_posts < median_posts:
            ptype = "niche_to_mainstream"
        elif first_mover_posts >= p75_posts:
            ptype = "top_down"
        else:
            ptype = "niche_to_mainstream"

        records.append({
            "event_cluster_id": int(row["event_cluster_id"]),
            "subreddit_sequence": "|".join(subreddit_seq),
            "propagation_type": ptype,
            "num_subreddits": int(row["num_subreddits"]),
            "total_duration_hours": float(row["total_duration_hours"]),
            "first_detection_time": row["first_detection_time"],
        })

    prop_pd = pd.DataFrame(records)
    prop_sdf = spark.createDataFrame(prop_pd)

    write_intermediate(prop_sdf, "propagation_events.parquet")
    log.info("Propagation events written: %d clusters", len(records))

    # =====================================================================
    # Visualizations
    # =====================================================================
    # --- 1. Propagation type distribution --------------------------------
    type_counts = prop_pd["propagation_type"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"niche_to_mainstream": "#3498DB", "simultaneous": "#2ECC71",
              "top_down": "#E74C3C"}
    bar_colors = [colors.get(t, "#888") for t in type_counts.index]
    ax.bar(type_counts.index, type_counts.values, color=bar_colors, edgecolor="white")
    ax.set_xlabel("Propagation Type")
    ax.set_ylabel("Number of Event Clusters")
    ax.set_title("Distribution of Propagation Types")
    for i, (t, v) in enumerate(type_counts.items()):
        ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
    save_fig(fig, "propagation_type_distribution.png")

    # --- 2. Scatter: time-offset vs subreddit-size for largest clusters --
    top_clusters = prop_pd.nlargest(10, "num_subreddits")["event_cluster_id"].tolist()
    scatter_data = aw_full.filter(
        F.col("event_cluster_id").isin(top_clusters)
    ).toPandas()

    if not scatter_data.empty:
        # Compute offset from first detection in each cluster
        first_times = scatter_data.groupby("event_cluster_id")["window_start"].min()
        scatter_data = scatter_data.merge(
            first_times.rename("cluster_first"), on="event_cluster_id"
        )
        scatter_data["offset_hours"] = (
            (scatter_data["window_start"] - scatter_data["cluster_first"])
            .dt.total_seconds() / 3600
        )

        fig, ax = plt.subplots(figsize=(12, 7))
        scatter = ax.scatter(
            scatter_data["offset_hours"],
            scatter_data["total_posts"].fillna(0),
            c=scatter_data["event_cluster_id"].astype("category").cat.codes,
            cmap="tab10", alpha=0.7, s=40, edgecolors="white", linewidth=0.5,
        )
        ax.set_xlabel("Hours After First Detection in Cluster")
        ax.set_ylabel("Subreddit Total Posts (size proxy)")
        ax.set_title("Propagation Timing vs Subreddit Size (Top 10 Clusters)")
        ax.set_yscale("log")
        save_fig(fig, "propagation_scatter.png")
    else:
        log.warning("No scatter data for top clusters.")

    # --- 3. Network graph of top 10 clusters -----------------------------
    try:
        import networkx as nx

        fig, axes = plt.subplots(2, 5, figsize=(24, 10))
        axes = axes.flatten()

        for idx, cid in enumerate(top_clusters[:10]):
            ax = axes[idx] if idx < 10 else None
            if ax is None:
                break

            cluster_wids = [
                wid for wid, c in wid_to_cluster.items() if c == cid
            ]
            subG = G.subgraph(cluster_wids).copy()

            # Relabel nodes from window_id to subreddit name
            wid_to_sub = dict(
                aw_full.filter(F.col("event_cluster_id") == cid)
                .select("window_id", "subreddit")
                .toPandas()
                .values
            )
            mapping = {wid: wid_to_sub.get(wid, wid) for wid in subG.nodes()}
            subG = nx.relabel_nodes(subG, mapping)

            pos = nx.spring_layout(subG, seed=42)
            nx.draw_networkx(
                subG, pos, ax=ax, node_size=300, node_color="#3498DB",
                font_size=7, font_weight="bold", edge_color="#bbb",
                with_labels=True,
            )
            ax.set_title(f"Cluster {cid} ({len(subG.nodes)}n)", fontsize=10)
            ax.axis("off")

        # Hide unused axes
        for j in range(len(top_clusters), 10):
            axes[j].axis("off")

        fig.suptitle("Network Graphs of Top 10 Event Clusters", fontsize=14)
        save_fig(fig, "propagation_network_top10.png")
    except Exception as exc:
        log.warning("Could not generate network visualization: %s", exc)

    # ----- Summary stats -------------------------------------------------
    print("\n--- Stage 3 Summary ---")
    print(f"  Total event clusters        : {len(records):,}")
    print(f"  Co-occurring anomaly pairs  : {len(co_occur_pd):,}")
    print(f"  Propagation types           :")
    for t, c in type_counts.items():
        print(f"    {t:25s}: {c}")
    if not prop_pd.empty:
        print(f"  Avg subreddits per cluster  : {prop_pd['num_subreddits'].mean():.1f}")
        print(f"  Avg cluster duration (hours): {prop_pd['total_duration_hours'].mean():.1f}")
    print(f"  Elapsed time                : {time.time() - t0:.1f}s")

    spark.stop()
    log.info("Stage 3 complete.")


if __name__ == "__main__":
    main()
