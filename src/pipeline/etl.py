"""Streaming Out-of-Core ETL pipeline using Polars.

Joins transactions with order metadata, enforces aggressive downcasting,
eliminates rare items and single-item baskets, and outputs compressed Parquet.
"""

import logging
import time
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ORDER_PRODUCTS_SCHEMA = {
    "order_id": pl.UInt32,
    "product_id": pl.UInt32,
    "add_to_cart_order": pl.UInt8,
    "reordered": pl.UInt8,
}

ORDERS_SCHEMA = {
    "order_id": pl.UInt32,
    "user_id": pl.UInt32,
    "order_number": pl.UInt8,
}


def build_interactions_pipeline(
    order_products_path: str | Path,
    orders_path: str | Path,
    min_basket_size: int = 2,
    min_product_frequency: int = 10,
) -> pl.LazyFrame:
    """Builds a lazy computation graph for Instacart transactions."""
    products_lf = pl.scan_csv(
        order_products_path,
        schema_overrides=ORDER_PRODUCTS_SCHEMA,
    )
    orders_lf = pl.scan_csv(
        orders_path,
        schema_overrides=ORDERS_SCHEMA,
    ).select(["order_id", "user_id", "order_number"])

    df = products_lf.join(orders_lf, on="order_id", how="inner")

    # ШАГ 1: удаляем редкие товары по глобальной частоте
    if min_product_frequency > 1:
        df = df.filter(pl.len().over("product_id") >= min_product_frequency)

    # ШАГ 2: удаляем корзины, деградировавшие после фильтрации товаров
    if min_basket_size > 1:
        df = df.filter(pl.len().over("order_id") >= min_basket_size)

    return df.select(
        [
            "order_id",
            "user_id",
            "product_id",
            "add_to_cart_order",
            "order_number",
            "reordered",
        ]
    )


def run_etl(
    order_products_path: Path | str,
    orders_path: Path | str,
    output_parquet_path: Path | str = "data/processed/interactions.parquet",
    sample_output_path: Path | str = "data/sample/interactions_sample.parquet",
    min_basket_size: int = 2,
    min_product_frequency: int = 10,
    sample_size: int = 1_000,
) -> None:
    """Executes the pipeline, saves target Parquet and generates a CI sample."""
    start_time = time.perf_counter()
    out_file = Path(output_parquet_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing lazy execution graph...")
    pipeline_lf = build_interactions_pipeline(
        order_products_path=order_products_path,
        orders_path=orders_path,
        min_basket_size=min_basket_size,
        min_product_frequency=min_product_frequency,
    )

    logger.info("Streaming dataset to %s...", out_file)
    pipeline_lf.sink_parquet(
        out_file,
        compression="zstd",
        compression_level=3,
        row_group_size=100_000,
    )

    elapsed_time = time.perf_counter() - start_time
    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    logger.info("ETL completed in %.2f s. File size: %.2f MB", elapsed_time, file_size_mb)

    # Генерация детерминированного сэмпла из готового Parquet без повторного чтения CSV
    if sample_output_path:
        sample_file = Path(sample_output_path)
        sample_file.parent.mkdir(parents=True, exist_ok=True)

        sample_df = pl.read_parquet(out_file, n_rows=min(sample_size, 1_000))
        sample_df.write_parquet(sample_file, compression="zstd")
        sample_size_kb = sample_file.stat().st_size / 1024
        logger.info(
            "Created CI sample at %s (%d rows, %.2f KB)",
            sample_file,
            len(sample_df),
            sample_size_kb,
        )


if __name__ == "__main__":
    run_etl(
        order_products_path="data/raw/order_products__prior.csv",
        orders_path="data/raw/orders.csv",
    )
