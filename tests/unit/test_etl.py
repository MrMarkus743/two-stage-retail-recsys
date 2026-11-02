"""Unit tests for the ETL pipeline, schema validation and cascading filters."""

from pathlib import Path

import polars as pl
import pytest

from src.pipeline.etl import build_interactions_pipeline, run_etl


@pytest.fixture
def synthetic_data(tmp_path: Path) -> tuple[Path, Path]:
    """Generates synthetic dataset to test edge-cases and cascading filters."""
    products_csv = tmp_path / "order_products.csv"
    orders_csv = tmp_path / "orders.csv"

    # 101 (3 раза), 102 (2 раза), 103 (2 раза) — частые товары.
    # 999 (1 раз) — редкий товар.
    # Заказ 1: [101, 102, 103] -> валидный (3 товара).
    # Заказ 2: [101] -> корзина < 2, отсекается.
    # Заказ 3: [101, 999] -> после удаления 999 останется 1 товар, деградирует и отсекается.
    # Заказы 4 и 5: корзины размера 1, отсекаются по min_basket_size, но дают частоту 102 и 103.
    products_csv.write_text(
        "order_id,product_id,add_to_cart_order,reordered\n"
        "1,101,1,0\n"
        "1,102,2,1\n"
        "1,103,3,0\n"
        "2,101,1,1\n"
        "3,101,1,0\n"
        "3,999,2,0\n"
        "4,102,1,0\n"
        "5,103,1,0\n"
    )

    orders_csv.write_text(
        "order_id,user_id,order_number\n1,501,1\n2,502,3\n3,501,2\n4,501,3\n5,502,1\n"
    )

    return products_csv, orders_csv


def test_etl_downcasting_schema(synthetic_data: tuple[Path, Path]) -> None:
    """Validates that schema downcasting applies UInt32 and UInt8 correctly."""
    products_csv, orders_csv = synthetic_data
    schema = build_interactions_pipeline(
        products_csv, orders_csv, min_basket_size=1, min_product_frequency=1
    ).collect_schema()

    assert schema["order_id"] == pl.UInt32
    assert schema["user_id"] == pl.UInt32
    assert schema["product_id"] == pl.UInt32
    assert schema["add_to_cart_order"] == pl.UInt8
    assert schema["order_number"] == pl.UInt8
    assert schema["reordered"] == pl.UInt8


def test_cascading_filter_removes_degraded_baskets(
    synthetic_data: tuple[Path, Path],
) -> None:
    """Verifies that baskets degrading to < 2 items after rare item removal are purged."""
    products_csv, orders_csv = synthetic_data

    # Порог частоты товара = 2, минимальный размер корзины = 2
    df = build_interactions_pipeline(
        products_csv, orders_csv, min_basket_size=2, min_product_frequency=2
    ).collect()

    # Товар 999 должен быть удален
    assert 999 not in df["product_id"].to_list()

    # Заказ 3 деградировал до 1 товара и отсекается
    assert 3 not in df["order_id"].to_list()

    # Остался только заказ 1
    assert df["order_id"].unique().to_list() == [1]


def test_run_etl_sample_creation(synthetic_data: tuple[Path, Path], tmp_path: Path) -> None:
    """Tests pipeline output files generation and sample sizing."""
    products_csv, orders_csv = synthetic_data
    out_parquet = tmp_path / "interactions.parquet"
    sample_parquet = tmp_path / "sample.parquet"

    run_etl(
        order_products_path=products_csv,
        orders_path=orders_csv,
        output_parquet_path=out_parquet,
        sample_output_path=sample_parquet,
        min_basket_size=2,
        min_product_frequency=1,
        sample_size=2,
    )

    assert out_parquet.exists()
    assert sample_parquet.exists()

    df_sample = pl.read_parquet(sample_parquet)
    assert len(df_sample) == 2
    assert df_sample["order_id"].dtype == pl.UInt32
