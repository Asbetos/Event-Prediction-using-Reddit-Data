"""Spark session factory tuned for EC2 t3.large (2 vCPU, 7.6 GB RAM)."""

import os
from pyspark.sql import SparkSession

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-17-openjdk-amd64")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_spark_session(app_name: str = "RedditEWS",
                         driver_memory: str = "4g") -> SparkSession:
    """Create a Spark session optimized for t3.large.

    Memory budget:
      JVM driver: 4 GB | JVM overhead: ~400 MB | Python: ~3 GB | OS: ~200 MB
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[2]")

        # Memory
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", "1g")

        # Shuffle / partitions
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "4")
        .config("spark.sql.files.maxPartitionBytes", "128m")

        # Adaptive Query Execution
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.minPartitionNum", "2")

        # Serialization
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

        # Arrow for pandas
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")

        # Spill to disk
        .config("spark.memory.fraction", "0.6")
        .config("spark.memory.storageFraction", "0.3")
        .config("spark.local.dir", os.path.join(PROJECT_DIR, "data", "spark-tmp"))

        # S3A access
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")

        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
