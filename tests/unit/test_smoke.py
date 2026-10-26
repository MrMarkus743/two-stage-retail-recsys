"""Smoke tests to verify test runner configuration and core library runtime."""

import numpy as np
import polars as pl
import scipy.sparse as sp


def test_environment_smoke() -> None:
    """Basic sanity check ensuring pytest runner executes properly."""
    assert 1 + 1 == 2


def test_core_dependencies_runtime() -> None:
    """Verify that computational and data processing libraries initialize without ABI errors."""
    # Polars DataFrame sanity check
    df = pl.DataFrame({"product_id": [1, 2, 3], "reordered": [0, 1, 1]})
    assert df.shape == (3, 2)
    assert df["reordered"].sum() == 2

    # NumPy array check
    arr = np.array([10, 20, 30], dtype=np.uint32)
    assert arr.sum() == 60

    # SciPy CSR sparse matrix initialization
    indptr = np.array([0, 2, 3, 6], dtype=np.int32)
    indices = np.array([0, 2, 2, 0, 1, 2], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    matrix = sp.csr_matrix((data, indices, indptr), shape=(3, 3))

    assert matrix.nnz == 6
    assert matrix.shape == (3, 3)
