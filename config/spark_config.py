"""Spark session factory tuned for EC2 t3.large (2 vCPU, 7.6 GB RAM)."""

import os
from pyspark.sql import SparkSession

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-17-openjdk-amd64")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_spark_session(app_name: str = "RedditEWS",
                         driver_memory: str = "3g") -> SparkSession:
    """Create a Spark session optimized for t3.large (2 vCPU, 7.6 GB RAM).

    Memory budget:
      JVM driver: 3 GB | JVM overhead: ~400 MB | Python: ~3.5 GB | OS: ~700 MB
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[2]")

        # Memory
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", "512m")

        # Shuffle / partitions — tuned for 2 cores
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "2")
        .config("spark.sql.files.maxPartitionBytes", "256m")

        # Adaptive Query Execution
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.minPartitionNum", "1")

        # Serialization
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

        # Arrow for pandas
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")

        # Memory fractions — lower JVM fraction leaves more for Python/pandas
        .config("spark.memory.fraction", "0.4")
        .config("spark.memory.storageFraction", "0.2")
        .config("spark.local.dir", os.path.join(PROJECT_DIR, "data", "spark-tmp"))

        # S3A access
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")

        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
