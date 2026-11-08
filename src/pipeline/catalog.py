"""Catalog metadata repository and Cold-Start Fallback recommender.

Joins product catalog with aisles and departments via Polars, aggregates
global bestseller rankings, and provides O(1) in-memory metadata access.
"""

import json
import logging
import time
from collections.abc import Collection
from pathlib import Path
from typing import Any

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_catalog_metadata(
    products_path: Path | str,
    aisles_path: Path | str,
    departments_path: Path | str,
) -> dict[int, dict[str, Any]]:
    """Joins products, aisles, and departments into an O(1) lookup dictionary.

    Returns:
        Mapping of product_id -> {product_name, aisle_id, department_id, aisle, department}.
    """
    products_df = pl.scan_csv(
        products_path,
        schema_overrides={
            "product_id": pl.UInt32,
            "product_name": pl.Utf8,
            "aisle_id": pl.UInt16,
            "department_id": pl.UInt16,
        },
    )

    aisles_df = pl.scan_csv(
        aisles_path,
        schema_overrides={"aisle_id": pl.UInt16, "aisle": pl.Utf8},
    )

    departments_df = pl.scan_csv(
        departments_path,
        schema_overrides={"department_id": pl.UInt16, "department": pl.Utf8},
    )

    catalog_df = (
        products_df.join(aisles_df, on="aisle_id", how="left")
        .join(departments_df, on="department_id", how="left")
        .collect()
    )

    catalog: dict[int, dict[str, Any]] = {}
    for row in catalog_df.iter_rows(named=True):
        catalog[int(row["product_id"])] = {
            "name": row["product_name"],
            "aisle_id": int(row["aisle_id"]),
            "department_id": int(row["department_id"]),
            "aisle": row["aisle"],
            "department": row["department"],
        }

    return catalog


def compute_top_popular_products(
    interactions_path: Path | str,
    top_n: int = 100,
) -> list[int]:
    """Calculates global bestsellers based on purchase frequency in interactions.

    Args:
        interactions_path: Path to processed interactions.parquet.
        top_n: Maximum number of top products to keep.

    Returns:
        List of product_ids sorted by popularity descending.
    """
    df = (
        pl.scan_parquet(interactions_path)
        .select("product_id")
        .group_by("product_id")
        .len(name="frequency")
        .sort("frequency", descending=True)
        .limit(top_n)
        .collect()
    )

    return [int(pid) for pid in df["product_id"].to_list()]


class CatalogRepository:
    """In-memory metadata store and fallback recommender."""

    def __init__(
        self,
        metadata: dict[int, dict[str, Any]],
        top_popular: list[int],
    ) -> None:
        self._metadata = metadata
        self._top_popular = top_popular

    def get_item(self, product_id: int) -> dict[str, Any] | None:
        """Retrieves item metadata in O(1) time."""
        return self._metadata.get(product_id)

    def get_fallback_recommendations(
        self,
        exclude_product_ids: Collection[int] | None = None,
        limit: int = 10,
    ) -> list[int]:
        """Provides non-duplicate bestseller recommendations for cold start scenarios.

        Args:
            exclude_product_ids: IDs already present in the user's cart.
            limit: Number of recommendations to return.

        Returns:
            List of recommended product IDs.
        """
        exclude_set = set(exclude_product_ids) if exclude_product_ids else set()
        recommendations: list[int] = []

        for pid in self._top_popular:
            if pid not in exclude_set:
                recommendations.append(pid)
            if len(recommendations) >= limit:
                break

        return recommendations


def run_catalog_pipeline(
    products_path: Path | str = "data/raw/products.csv",
    aisles_path: Path | str = "data/raw/aisles.csv",
    departments_path: Path | str = "data/raw/departments.csv",
    interactions_path: Path | str = "data/processed/interactions.parquet",
    output_catalog_path: Path | str = "data/processed/catalog_metadata.json",
    output_popular_path: Path | str = "data/processed/top_popular.json",
) -> None:
    """Executes catalog join, popularity ranking calculation, and serializes artifacts."""
    start_time = time.perf_counter()
    out_cat = Path(output_catalog_path)
    out_pop = Path(output_popular_path)

    out_cat.parent.mkdir(parents=True, exist_ok=True)
    out_pop.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building metadata catalog from raw CSVs...")
    catalog = build_catalog_metadata(products_path, aisles_path, departments_path)

    logger.info("Serializing catalog metadata to %s...", out_cat)
    with open(out_cat, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False)

    logger.info("Calculating Top-100 popular products from %s...", interactions_path)
    top_popular = compute_top_popular_products(interactions_path, top_n=100)

    logger.info("Serializing top popular products to %s...", out_pop)
    with open(out_pop, "w", encoding="utf-8") as f:
        json.dump({"top_popular": top_popular}, f, ensure_ascii=False)

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Catalog pipeline completed in %.2f s. Items: %d, Bestsellers: %d",
        elapsed,
        len(catalog),
        len(top_popular),
    )


if __name__ == "__main__":
    run_catalog_pipeline()
