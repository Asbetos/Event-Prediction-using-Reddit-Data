#!/usr/bin/env python3
"""
Spark MLlib Demonstration -- Event Classification & Sustain/Decay Prediction
=============================================================================
Supplementary stage that demonstrates Spark MLlib usage alongside the primary
GPU-based cuML results.  Assembles features from prior stages, trains a
RandomForestClassifier for event-category classification (Q8) and uses
BinaryClassificationEvaluator for sustain/decay prediction (Q9).

This satisfies the course requirement for Spark MLlib utilization.
"""

import os, sys, logging, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, IndexToString
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)

from config.spark_config import create_spark_session
from config.settings import LOCAL_GROUND_TRUTH
from utils.spark_utils import read_intermediate

import pandas as pd

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("stage_ml")


def load_ground_truth(spark):
    """Load ground truth events and explode subreddits."""
    gt_path = os.path.join(LOCAL_GROUND_TRUTH, "events.csv")
    if not os.path.exists(gt_path):
        return None
    gt = (
        spark.read.csv(gt_path, header=True, inferSchema=True)
        .withColumn("date", F.to_date("date", "yyyy-MM-dd"))
        .withColumn("subreddits_arr", F.split(F.col("relevant_subreddits"), "\\|"))
    )
    return gt.select(
        "event_id", "date", "name", "category",
        F.explode("subreddits_arr").alias("subreddit"),
    )


def build_feature_matrix(spark):
    """
    Assemble a feature matrix by joining outputs from Stages 2-5.

    Returns a Spark DataFrame with numeric features and a 'category' label
    (from ground truth), plus a binary 'sustained' label for Q9.
    """
    # --- Load intermediate artifacts ------------------------------------
    try:
        anomaly_windows = read_intermediate(spark, "anomaly_windows.parquet")
    except Exception:
        log.error("anomaly_windows.parquet not found. Run Stage 2 first.")
        return None

    spike_profiles = None
    try:
        spike_profiles = read_intermediate(spark, "spike_profiles.parquet")
    except Exception:
        log.info("spike_profiles.parquet not found; features will be limited.")

    temporal_patterns = None
    try:
        temporal_patterns = read_intermediate(spark, "temporal_patterns.parquet")
    except Exception:
        log.info("temporal_patterns.parquet not found; features will be limited.")

    # --- Base feature set from anomaly windows --------------------------
    features = anomaly_windows.select(
        "window_id",
        "subreddit",
        "window_start",
        F.col("peak_z_score").cast(DoubleType()),
        F.col("mean_z_score").cast(DoubleType()),
        F.col("duration_hours").cast(DoubleType()),
        F.col("anomaly_hours").cast(DoubleType()),
        F.col("peak_post_count").cast(DoubleType()),
        F.col("mean_post_count").cast(DoubleType()),
    )

    # Add temporal features
    features = features.withColumn(
        "hour_of_day", F.hour("window_start").cast(DoubleType())
    ).withColumn(
        "day_of_week",
        # Spark dayofweek: 1=Sun,2=Mon,...,7=Sat -> remap to 0=Mon..6=Sun
        ((F.dayofweek("window_start") + 5) % 7).cast(DoubleType()),
    ).withColumn(
        "is_weekend",
        F.when(F.col("day_of_week").isin(5.0, 6.0), 1.0).otherwise(0.0),
    )

    # Add spike profile features if available
    if spike_profiles is not None:
        spike_feats = spike_profiles.select(
            "window_id",
            F.col("post_spike_avg_score").cast(DoubleType()),
            F.col("post_spike_unique_authors").cast(DoubleType()),
            F.col("baseline_avg_score").cast(DoubleType()),
            F.col("engagement_ratio").cast(DoubleType()),
        )
        features = features.join(spike_feats, on="window_id", how="left")

    # --- Join with ground truth for labels ------------------------------
    gt = load_ground_truth(spark)
    if gt is None:
        log.error("Ground truth not found. Cannot build labeled dataset.")
        return None

    # Match anomaly windows to events by subreddit + date overlap
    features = features.withColumn(
        "window_date", F.to_date("window_start")
    )
    gt = gt.withColumn("gt_date", F.col("date"))

    labeled = features.join(
        gt,
        on=[
            features["subreddit"] == gt["subreddit"],
            F.abs(F.datediff(features["window_date"], gt["gt_date"])) <= 2,
        ],
        how="inner",
    ).drop(gt["subreddit"])

    if labeled.count() == 0:
        log.warning("No anomaly windows matched to ground truth events.")
        return None

    # Binary label: sustained = duration > 12h (proxy for Q9)
    labeled = labeled.withColumn(
        "sustained",
        F.when(F.col("duration_hours") > 12, 1.0).otherwise(0.0),
    )

    log.info("Labeled feature matrix: %d rows", labeled.count())
    return labeled


def run_event_classification(labeled):
    """Q8: Multi-class event category classification via RandomForest."""
    print("\n" + "=" * 70)
    print("  Q8: EVENT CATEGORY CLASSIFICATION (Spark MLlib RandomForest)")
    print("=" * 70)

    feature_cols = [
        "peak_z_score", "mean_z_score", "duration_hours",
        "anomaly_hours", "peak_post_count", "mean_post_count",
        "hour_of_day", "day_of_week", "is_weekend",
    ]

    # Add spike features if present
    for col in ["post_spike_avg_score", "post_spike_unique_authors",
                "baseline_avg_score", "engagement_ratio"]:
        if col in labeled.columns:
            feature_cols.append(col)

    # Drop rows with null features
    clean = labeled.dropna(subset=feature_cols + ["category"])
    log.info("Clean rows for classification: %d", clean.count())

    if clean.count() < 10:
        log.warning("Too few labeled samples for meaningful classification.")
        return

    # Pipeline
    indexer = StringIndexer(inputCol="category", outputCol="label", handleInvalid="skip")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features",
                                handleInvalid="skip")
    rf = RandomForestClassifier(
        labelCol="label", featuresCol="features",
        numTrees=50, maxDepth=8, seed=42,
    )
    label_converter = IndexToString(
        inputCol="prediction", outputCol="predicted_category",
        labels=indexer.fit(clean).labels,
    )

    pipeline = Pipeline(stages=[indexer, assembler, rf, label_converter])

    # Train / test split
    train, test = clean.randomSplit([0.7, 0.3], seed=42)
    log.info("Train: %d, Test: %d", train.count(), test.count())

    model = pipeline.fit(train)
    predictions = model.transform(test)

    # Evaluation
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    evaluator_wp = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
    )
    evaluator_wr = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"
    )

    accuracy = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    precision = evaluator_wp.evaluate(predictions)
    recall = evaluator_wr.evaluate(predictions)

    print(f"\n  Classification Results (test set):")
    print(f"    Accuracy           : {accuracy:.4f}")
    print(f"    Weighted F1        : {f1:.4f}")
    print(f"    Weighted Precision : {precision:.4f}")
    print(f"    Weighted Recall    : {recall:.4f}")

    # Confusion matrix (small dataset so toPandas is fine)
    cm = (
        predictions
        .groupBy("category", "predicted_category")
        .count()
        .orderBy("category", "predicted_category")
        .toPandas()
    )
    print("\n  Confusion Matrix:")
    print(cm.to_string(index=False))

    # Feature importance
    rf_model = model.stages[2]
    importances = rf_model.featureImportances.toArray()
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    print("\n  Feature Importances:")
    print(fi_df.to_string(index=False))

    return model


def run_sustain_decay_prediction(labeled):
    """Q9: Binary classification -- sustained vs decaying anomaly."""
    print("\n" + "=" * 70)
    print("  Q9: SUSTAIN / DECAY PREDICTION (Spark MLlib BinaryClassification)")
    print("=" * 70)

    feature_cols = [
        "peak_z_score", "mean_z_score",
        "peak_post_count", "mean_post_count",
        "hour_of_day", "day_of_week", "is_weekend",
    ]
    for col in ["post_spike_avg_score", "post_spike_unique_authors",
                "baseline_avg_score", "engagement_ratio"]:
        if col in labeled.columns:
            feature_cols.append(col)

    clean = labeled.dropna(subset=feature_cols + ["sustained"])
    log.info("Clean rows for binary prediction: %d", clean.count())

    if clean.count() < 10:
        log.warning("Too few labeled samples for binary classification.")
        return

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features",
                                handleInvalid="skip")
    rf = RandomForestClassifier(
        labelCol="sustained", featuresCol="features",
        numTrees=50, maxDepth=6, seed=42,
    )
    pipeline = Pipeline(stages=[assembler, rf])

    train, test = clean.randomSplit([0.7, 0.3], seed=42)
    model = pipeline.fit(train)
    predictions = model.transform(test)

    # Binary evaluation
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="sustained", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="sustained", rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
    )

    auc_roc = evaluator_auc.evaluate(predictions)
    auc_pr = evaluator_pr.evaluate(predictions)

    # Also compute accuracy
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol="sustained", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = acc_evaluator.evaluate(predictions)

    print(f"\n  Binary Classification Results (test set):")
    print(f"    AUC-ROC  : {auc_roc:.4f}")
    print(f"    AUC-PR   : {auc_pr:.4f}")
    print(f"    Accuracy : {accuracy:.4f}")

    # Class distribution
    dist = (
        predictions
        .groupBy("sustained", "prediction")
        .count()
        .orderBy("sustained", "prediction")
        .toPandas()
    )
    print("\n  Prediction Distribution:")
    print(dist.to_string(index=False))

    # Feature importance
    rf_model = model.stages[1]
    importances = rf_model.featureImportances.toArray()
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    print("\n  Feature Importances:")
    print(fi_df.to_string(index=False))

    return model


def main():
    t0 = time.time()
    spark = create_spark_session(app_name="Stage_ML_Spark")
    log.info("Spark session created.")

    # ----- 1. Build feature matrix ---------------------------------------
    labeled = build_feature_matrix(spark)
    if labeled is None:
        log.error("Could not build feature matrix. Exiting.")
        spark.stop()
        sys.exit(1)

    # Cache for reuse
    labeled.cache()
    total = labeled.count()
    log.info("Feature matrix: %d rows", total)

    # Show label distribution
    print("\n  Category distribution:")
    labeled.groupBy("category").count().orderBy("category").show(truncate=False)

    print("\n  Sustained label distribution:")
    labeled.groupBy("sustained").count().show()

    # ----- 2. Q8: Event classification -----------------------------------
    try:
        run_event_classification(labeled)
    except Exception as exc:
        log.error("Event classification failed: %s", exc, exc_info=True)

    # ----- 3. Q9: Sustain/decay binary classification --------------------
    try:
        run_sustain_decay_prediction(labeled)
    except Exception as exc:
        log.error("Sustain/decay prediction failed: %s", exc, exc_info=True)

    # ----- Summary -------------------------------------------------------
    print(f"\n--- Spark MLlib Demo Summary ---")
    print(f"  Total labeled samples   : {total:,}")
    print(f"  Features used           : anomaly metrics + temporal + engagement")
    print(f"  Models                  : RandomForestClassifier (Q8 & Q9)")
    print(f"  Elapsed time            : {time.time() - t0:.1f}s")

    labeled.unpersist()
    spark.stop()
    log.info("Spark MLlib demo complete.")


if __name__ == "__main__":
    main()
