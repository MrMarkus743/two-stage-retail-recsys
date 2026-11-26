"""Vector similarity metrics for sparse CSR matrices (Cosine, Jaccard).

Computes vector similarities directly in sparse representation without
dense array unpacking to maintain minimal memory footprint and high throughput.
"""

import numpy as np
import scipy.sparse as sp


def normalize_l2_sparse(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Performs vectorized row-wise L2-normalization of a CSR matrix.

    Args:
        matrix: Input scipy.sparse.csr_matrix.

    Returns:
        L2-normalized scipy.sparse.csr_matrix (all-zero rows remain zero).
    """
    if not isinstance(matrix, sp.csr_matrix):
        matrix = sp.csr_matrix(matrix)

    mat = matrix.copy()
    mat.eliminate_zeros()

    # Считаем сумму квадратов элементов в строках напрямую в разреженном виде
    row_sum_sq = np.array(mat.multiply(mat).sum(axis=1)).ravel()
    row_norms = np.sqrt(row_sum_sq)

    # Защита от деления на ноль для пустых строк
    inv_norms = np.zeros_like(row_norms, dtype=np.float32)
    non_zero = row_norms > 0
    inv_norms[non_zero] = 1.0 / row_norms[non_zero]

    # Векторизованное масштабирование строк через левое умножение на диагональную матрицу
    scaler = sp.diags(
        inv_norms,
        offsets=0,
        shape=(mat.shape[0], mat.shape[0]),
        format="csr",
    )
    normalized = (scaler @ mat).tocsr()
    normalized.eliminate_zeros()
    return normalized.astype(np.float32)


def cosine_similarity_sparse(
    u: sp.csr_matrix,
    v: sp.csr_matrix,
    normalized: bool = False,
) -> float | np.ndarray:
    """Computes cosine similarity between sparse CSR vectors or matrices.

    Args:
        u: Sparse CSR vector (1, D) or matrix (N, D).
        v: Sparse CSR vector (1, D) or matrix (M, D).
        normalized: If True, assumes inputs are already unit-norm (L2-normalized).

    Returns:
        Float score for (1, D) vs (1, D), 1D numpy array for (1, D) vs (M, D),
        or 2D numpy array for general matrix dot product.
    """
    if not isinstance(u, sp.csr_matrix):
        u = sp.csr_matrix(u)
    if not isinstance(v, sp.csr_matrix):
        v = sp.csr_matrix(v)

    if u.shape[1] != v.shape[1]:
        raise ValueError(f"Dimension mismatch: u has {u.shape[1]} features, v has {v.shape[1]}")

    u_norm = u if normalized else normalize_l2_sparse(u)
    v_norm = v if normalized else normalize_l2_sparse(v)

    # Скалярное произведение L2-нормализованных векторов
    sim_matrix = (u_norm @ v_norm.T).toarray()
    np.clip(sim_matrix, -1.0, 1.0, out=sim_matrix)

    if u.shape[0] == 1 and v.shape[0] == 1:
        return float(sim_matrix[0, 0])
    if u.shape[0] == 1:
        return sim_matrix[0]
    return sim_matrix


def jaccard_similarity_sparse(
    u: sp.csr_matrix,
    v: sp.csr_matrix,
) -> float | np.ndarray:
    """Computes Jaccard similarity index for binary interaction representations.

    J(u, v) = |u ∩ v| / |u ∪ v| = |u ∩ v| / (|u| + |v| - |u ∩ v|).

    Args:
        u: Sparse CSR vector (1, D) or matrix (N, D).
        v: Sparse CSR vector (1, D) or matrix (M, D).

    Returns:
        Float Jaccard index in [0.0, 1.0] for single rows, or numpy array.
    """
    if not isinstance(u, sp.csr_matrix):
        u = sp.csr_matrix(u)
    if not isinstance(v, sp.csr_matrix):
        v = sp.csr_matrix(v)

    if u.shape[1] != v.shape[1]:
        raise ValueError(f"Dimension mismatch: u has {u.shape[1]} features, v has {v.shape[1]}")

    u_bin = u.copy()
    v_bin = v.copy()
    u_bin.eliminate_zeros()
    v_bin.eliminate_zeros()

    # Бинаризация ненулевых весов
    u_bin.data = np.ones_like(u_bin.data, dtype=np.float32)
    v_bin.data = np.ones_like(v_bin.data, dtype=np.float32)

    # Пересечение: число общих ненулевых позиций через скалярное произведение
    intersection = (u_bin @ v_bin.T).toarray()

    # Мощности множеств |u| и |v| через дифференциалы indptr
    u_sizes = np.diff(u_bin.indptr)[:, np.newaxis]
    v_sizes = np.diff(v_bin.indptr)[np.newaxis, :]

    union = u_sizes + v_sizes - intersection

    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, intersection / union, 0.0).astype(np.float32)

    if u.shape[0] == 1 and v.shape[0] == 1:
        return float(jaccard[0, 0])
    if u.shape[0] == 1:
        return jaccard[0]
    return jaccard
