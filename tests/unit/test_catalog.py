"""Unit tests for catalog metadata repository and cold start fallback logic."""

from pathlib import Path

import polars as pl
import pytest

from src.pipeline.catalog import (
    CatalogRepository,
    build_catalog_metadata,
    compute_top_popular_products,
)


@pytest.fixture
def synthetic_catalog_csvs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Creates synthetic products, aisles, and departments CSVs."""
    products_csv = tmp_path / "products.csv"
    aisles_csv = tmp_path / "aisles.csv"
    departments_csv = tmp_path / "departments.csv"

    departments_csv.write_text("department_id,department\n1,produce\n2,dairy eggs\n")
    aisles_csv.write_text("aisle_id,aisle\n10,fresh fruits\n20,milk\n")
    products_csv.write_text(
        "product_id,product_name,aisle_id,department_id\n"
        "101,Organic Banana,10,1\n"
        "102,Whole Milk,20,2\n"
        "103,Organic Strawberries,10,1\n"
    )

    return products_csv, aisles_csv, departments_csv


@pytest.fixture
def synthetic_interactions_parquet(tmp_path: Path) -> Path:
    """Creates synthetic interactions parquet with known frequencies."""
    parquet_path = tmp_path / "interactions.parquet"
    # Товар 102 куплен 3 раза, 101 - 2 раза, 103 - 1 раз
    df = pl.DataFrame(
        {
            "order_id": [1, 1, 2, 2, 3, 3],
            "product_id": [102, 101, 102, 103, 102, 101],
        },
        schema={"order_id": pl.UInt32, "product_id": pl.UInt32},
    )
    df.write_parquet(parquet_path)
    return parquet_path


def test_build_catalog_metadata_mapping(
    synthetic_catalog_csvs: tuple[Path, Path, Path],
) -> None:
    """Tests O(1) dictionary creation with full category enrichment."""
    p_csv, a_csv, d_csv = synthetic_catalog_csvs
    catalog = build_catalog_metadata(p_csv, a_csv, d_csv)

    assert len(catalog) == 3
    assert 101 in catalog

    banana = catalog[101]
    assert banana["name"] == "Organic Banana"
    assert banana["aisle_id"] == 10
    assert banana["department_id"] == 1
    assert banana["aisle"] == "fresh fruits"
    assert banana["department"] == "produce"


def test_compute_top_popular_products(synthetic_interactions_parquet: Path) -> None:
    """Verifies that bestsellers are sorted strictly descending by purchase count."""
    top_popular = compute_top_popular_products(synthetic_interactions_parquet, top_n=2)

    # 102 (3 раза) -> 101 (2 раза)
    assert top_popular == [102, 101]


def test_catalog_repository_fallback_deduplication() -> None:
    """Validates that fallback does not recommend products already in cart."""
    mock_metadata = {
        101: {"name": "Banana", "aisle_id": 10},
        102: {"name": "Milk", "aisle_id": 20},
        103: {"name": "Bread", "aisle_id": 30},
        104: {"name": "Eggs", "aisle_id": 20},
    }
    mock_popular = [101, 102, 103, 104]

    repo = CatalogRepository(metadata=mock_metadata, top_popular=mock_popular)

    # 1. Запрос при пустой корзине
    assert repo.get_fallback_recommendations(exclude_product_ids=[], limit=2) == [
        101,
        102,
    ]

    # 2. Запрос с исключением: товар 101 уже в корзине
    recs = repo.get_fallback_recommendations(exclude_product_ids=[101], limit=2)
    assert 101 not in recs
    assert recs == [102, 103]

    # 3. Запрос с исключением нескольких товаров
    recs_multi = repo.get_fallback_recommendations(exclude_product_ids={101, 102}, limit=2)
    assert recs_multi == [103, 104]


def test_catalog_repository_metadata_lookup() -> None:
    """Verifies O(1) item retrieval and graceful None on unknown product."""
    repo = CatalogRepository(metadata={101: {"name": "Apple"}}, top_popular=[101])

    assert repo.get_item(101) == {"name": "Apple"}
    assert repo.get_item(9999) is None
