"""Unit tests for the CSR co-occurrence matrix builder."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.pipeline.matrix_builder import (
    build_cooccurrence_matrix,
    build_index_mappings,
    run_matrix_builder,
)


@pytest.fixture
def synthetic_sample_df() -> pl.DataFrame:
    """Creates a deterministic synthetic interaction table."""
    return pl.DataFrame(
        {
            "order_id": [1, 1, 1, 2, 2, 3, 3],
            "product_id": [101, 102, 103, 101, 102, 102, 104],
        },
        schema={"order_id": pl.UInt32, "product_id": pl.UInt32},
    )


def test_build_index_mappings(synthetic_sample_df: pl.DataFrame) -> None:
    """Verifies bijection and coverage of bidirectional index maps."""
    item_to_idx, idx_to_item = build_index_mappings(synthetic_sample_df)

    assert len(item_to_idx) == 4
    assert len(idx_to_item) == 4
    for pid, idx in item_to_idx.items():
        assert idx_to_item[idx] == pid


def test_cooccurrence_matrix_structure(synthetic_sample_df: pl.DataFrame) -> None:
    """Tests dimensions, symmetry, zeroed diagonal, and float32 dtype."""
    item_to_idx, _ = build_index_mappings(synthetic_sample_df)
    matrix = build_cooccurrence_matrix(synthetic_sample_df, item_to_idx)

    n_items = len(item_to_idx)
    assert matrix.shape == (n_items, n_items)
    assert matrix.dtype == np.float32

    # Проверка обнуления главной диагонали
    diagonal_elements = matrix.diagonal()
    assert np.all(diagonal_elements == 0.0)

    # Проверка строгой симметрии (C == C^T)
    diff = matrix - matrix.T
    assert diff.nnz == 0


def test_cooccurrence_values_and_retrieval(synthetic_sample_df: pl.DataFrame) -> None:
    """Validates exact co-occurrence count and neighborhood extraction."""
    item_to_idx, idx_to_item = build_index_mappings(synthetic_sample_df)
    matrix = build_cooccurrence_matrix(synthetic_sample_df, item_to_idx)

    idx_101 = item_to_idx[101]
    idx_102 = item_to_idx[102]

    # Товары 101 и 102 вместе были в заказах 1 и 2 -> вес должен быть равен 2.0
    assert matrix[idx_101, idx_102] == 2.0
    assert matrix[idx_102, idx_101] == 2.0

    # Извлечение вектора смежности для товара 101
    row_vector = matrix.getrow(idx_101).toarray().flatten()
    assert row_vector[idx_102] == 2.0
    assert row_vector[idx_101] == 0.0  # сам с собой обнулен


def test_run_matrix_builder_pipeline(tmp_path: Path, synthetic_sample_df: pl.DataFrame) -> None:
    """Validates full artifact serialization to disk."""
    parquet_path = tmp_path / "interactions.parquet"
    synthetic_sample_df.write_parquet(parquet_path)

    out_matrix = tmp_path / "matrix.npz"
    out_indices = tmp_path / "item_indices.json"

    run_matrix_builder(
        interactions_path=parquet_path,
        output_matrix_path=out_matrix,
        output_indices_path=out_indices,
    )

    assert out_matrix.exists()
    assert out_indices.exists()
