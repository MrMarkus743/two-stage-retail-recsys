"""Unit tests for sparse vector similarity metrics."""

import numpy as np
import pytest
import scipy.sparse as sp

from src.core.similarity import (
    cosine_similarity_sparse,
    jaccard_similarity_sparse,
    normalize_l2_sparse,
)


def test_normalize_l2_sparse() -> None:
    """Verifies that non-zero rows have unit Euclidean norm and zero rows stay zero."""
    matrix = sp.csr_matrix([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    normalized = normalize_l2_sparse(matrix)

    # Первая строка: 3^2 + 4^2 = 25 -> норма 5 -> [0.6, 0.8, 0.0]
    row0 = normalized.getrow(0).toarray().flatten()
    assert np.linalg.norm(row0) == pytest.approx(1.0)
    assert row0[0] == pytest.approx(0.6)
    assert row0[1] == pytest.approx(0.8)

    # Вторая строка (все нули): норма 0
    row1 = normalized.getrow(1).toarray().flatten()
    assert np.all(row1 == 0.0)

    # Третья строка: норма 1.0
    row2 = normalized.getrow(2).toarray().flatten()
    assert np.linalg.norm(row2) == pytest.approx(1.0)


def test_self_similarity_is_one() -> None:
    """Identical non-zero vectors must yield similarity strictly equal to 1.0."""
    v = sp.csr_matrix([[1.0, 5.0, 0.0, 2.0]])

    assert cosine_similarity_sparse(v, v) == pytest.approx(1.0)
    assert jaccard_similarity_sparse(v, v) == pytest.approx(1.0)


def test_orthogonal_vectors_yield_zero() -> None:
    """Orthogonal sparse vectors with disjoint indices must yield 0.0."""
    v1 = sp.csr_matrix([[1.0, 2.0, 0.0, 0.0]])
    v2 = sp.csr_matrix([[0.0, 0.0, 3.0, 4.0]])

    assert cosine_similarity_sparse(v1, v2) == 0.0
    assert jaccard_similarity_sparse(v1, v2) == 0.0


def test_zero_vector_edge_cases() -> None:
    """Comparisons with all-zero vectors must safely return 0.0 without division by zero."""
    zero_vec = sp.csr_matrix([[0.0, 0.0, 0.0]])
    normal_vec = sp.csr_matrix([[1.0, 2.0, 0.0]])

    assert cosine_similarity_sparse(zero_vec, normal_vec) == 0.0
    assert cosine_similarity_sparse(zero_vec, zero_vec) == 0.0
    assert jaccard_similarity_sparse(zero_vec, normal_vec) == 0.0
    assert jaccard_similarity_sparse(zero_vec, zero_vec) == 0.0


def test_jaccard_partial_overlap() -> None:
    """Validates mathematical correctness of Jaccard on binary overlapping sets."""
    # Множество 1: {0, 1} (размер 2)
    # Множество 2: {1, 2} (размер 2)
    # Пересечение: {1} (размер 1), Объединение: {0, 1, 2} (размер 3) -> J = 1/3
    v1 = sp.csr_matrix([[1.0, 1.0, 0.0]])
    v2 = sp.csr_matrix([[0.0, 1.0, 1.0]])

    assert jaccard_similarity_sparse(v1, v2) == pytest.approx(1.0 / 3.0)


def test_vector_to_matrix_broadcast() -> None:
    """Tests 1-to-M candidate evaluation returning a 1D score vector."""
    query = sp.csr_matrix([[1.0, 0.0]])
    candidates = sp.csr_matrix(
        [
            [1.0, 0.0],  # точное совпадение -> 1.0
            [0.0, 1.0],  # ортогонально -> 0.0
            [1.0, 1.0],  # пересечение
        ]
    )

    cos_scores = cosine_similarity_sparse(query, candidates)
    jacc_scores = jaccard_similarity_sparse(query, candidates)

    assert isinstance(cos_scores, np.ndarray)
    assert len(cos_scores) == 3
    assert cos_scores[0] == pytest.approx(1.0)
    assert cos_scores[1] == pytest.approx(0.0)

    assert isinstance(jacc_scores, np.ndarray)
    assert len(jacc_scores) == 3
    assert jacc_scores[0] == pytest.approx(1.0)
    assert jacc_scores[1] == pytest.approx(0.0)
    assert jacc_scores[2] == pytest.approx(0.5)  # 1 / (1 + 2 - 1) = 0.5


def test_dimension_mismatch_raises_value_error() -> None:
    """Verifies that vector dimension mismatch raises ValueError."""
    v1 = sp.csr_matrix([[1.0, 2.0]])
    v2 = sp.csr_matrix([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="Dimension mismatch"):
        cosine_similarity_sparse(v1, v2)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        jaccard_similarity_sparse(v1, v2)
