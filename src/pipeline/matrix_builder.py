"""Module for building and serializing the item-item co-occurrence CSR matrix.

Computes pairwise co-occurrence frequencies from interactions using optimized
bipartite incidence matrix multiplication to eliminate memory overhead.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def build_index_mappings(
    df: pl.DataFrame | pl.LazyFrame,
) -> tuple[dict[int, int], dict[int, int]]:
    """Builds bidirectional mappings between product_id and zero-based matrix indices.

    Args:
        df: DataFrame or LazyFrame containing a 'product_id' column.

    Returns:
        Tuple of (item_to_idx, idx_to_item).
    """
    if isinstance(df, pl.LazyFrame):
        df = df.select("product_id").unique().collect()

    unique_products = df.select("product_id").unique().sort("product_id")
    product_ids: list[int] = unique_products["product_id"].to_list()

    item_to_idx = {int(pid): idx for idx, pid in enumerate(product_ids)}
    idx_to_item = {idx: int(pid) for idx, pid in enumerate(product_ids)}

    return item_to_idx, idx_to_item


def build_cooccurrence_matrix(
    df: pl.DataFrame,
    item_to_idx: dict[int, int],
) -> sp.csr_matrix:
    """Builds a symmetric item co-occurrence CSR matrix with zeroed diagonal.

    Uses order-product bipartite representation.

    Args:
        df: Polars DataFrame with 'order_id' and 'product_id'.
        item_to_idx: Mapping from product_id to contiguous index.

    Returns:
        Symmetric scipy.sparse.csr_matrix of shape (N, N) with float32 values.
    """
    n_items = len(item_to_idx)

    # 1. Формируем непрерывные 0-based индексы для заказов через Polars
    orders_df = (
        df.select("order_id").unique().with_row_index("order_idx")
    )
    n_orders = len(orders_df)

    # 2. Быстрый маппинг индексов через join
    items_df = pl.DataFrame(
        {
            "product_id": list(item_to_idx.keys()),
            "item_idx": list(item_to_idx.values()),
        },
        schema={"product_id": pl.UInt32, "item_idx": pl.UInt32},
    )

    indexed_df = (
        df.select(["order_id", "product_id"])
        .join(orders_df, on="order_id", how="inner")
        .join(items_df, on="product_id", how="inner")
    )

    row_indices = indexed_df["order_idx"].to_numpy()
    col_indices = indexed_df["item_idx"].to_numpy()
    values = np.ones(len(indexed_df), dtype=np.float32)

    # 3. Создаем матрицу инцидентности A (Orders x Items)
    incidence_matrix = sp.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_orders, n_items),
        dtype=np.float32,
    )

    # 4. Расчет сопокупок
    cooc_matrix = (incidence_matrix.T @ incidence_matrix).tocsr()

    # 5. Исключение самосопокупок (обнуление главной диагонали)
    cooc_matrix.setdiag(0.0)
    cooc_matrix.eliminate_zeros()

    return cooc_matrix.astype(np.float32)


def run_matrix_builder(
    interactions_path: Path | str = "data/processed/interactions.parquet",
    output_matrix_path: Path | str = "data/processed/matrix.npz",
    output_indices_path: Path | str = "data/processed/item_indices.json",
) -> None:
    """End-to-end pipeline to generate matrix artifacts and mappings."""
    start_time = time.perf_counter()
    in_file = Path(interactions_path)
    matrix_file = Path(output_matrix_path)
    indices_file = Path(output_indices_path)

    matrix_file.parent.mkdir(parents=True, exist_ok=True)
    indices_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading interactions from %s...", in_file)
    df = pl.read_parquet(in_file, columns=["order_id", "product_id"])

    logger.info("Generating bidirectional index mappings...")
    item_to_idx, idx_to_item = build_index_mappings(df)

    logger.info("Building CSR co-occurrence matrix for %d items...", len(item_to_idx))
    matrix = build_cooccurrence_matrix(df, item_to_idx)

    logger.info("Saving CSR matrix to %s...", matrix_file)
    sp.save_npz(matrix_file, matrix)

    logger.info("Saving index mappings to %s...", indices_file)
    indices_payload = {
        "item_to_idx": item_to_idx,
        "idx_to_item": {str(k): v for k, v in idx_to_item.items()},
    }
    with open(indices_file, "w", encoding="utf-8") as f:
        json.dump(indices_payload, f, ensure_ascii=False)

    elapsed = time.perf_counter() - start_time
    matrix_size_mb = matrix_file.stat().st_size / (1024 * 1024)
    logger.info(
        "Matrix build completed in %.2f s. Matrix shape: %s, Non-zeros: %d, Size: %.2f MB",
        elapsed,
        matrix.shape,
        matrix.nnz,
        matrix_size_mb,
    )


if __name__ == "__main__":
    run_matrix_builder()
