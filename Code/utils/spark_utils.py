"""Shared Spark helpers."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.spark_config import create_spark_session
from config.settings import S3A_INTERMEDIATE, LOCAL_INTERMEDIATE


def read_intermediate(spark, name: str):
    """Read an intermediate parquet, preferring local then S3."""
    local = os.path.join(LOCAL_INTERMEDIATE, name)
    if os.path.exists(local):
        return spark.read.parquet(local)
    return spark.read.parquet(f"{S3A_INTERMEDIATE}/{name}")


def write_intermediate(df, name: str, mode: str = "overwrite"):
    """Write a Spark DataFrame as local parquet."""
    path = os.path.join(LOCAL_INTERMEDIATE, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.write.mode(mode).parquet(path)
    print(f"  Wrote {path}")
