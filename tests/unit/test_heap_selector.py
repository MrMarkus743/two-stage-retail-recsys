"""Unit tests for TopKHeapSelector verifying invariants and edge cases."""

import random

import pytest

from src.core.heap_selector import HeapEntry, TopKHeapSelector


def test_invalid_capacity() -> None:
    """Verifies that non-positive capacity raises ValueError."""
    with pytest.raises(ValueError, match="must be a positive integer"):
        TopKHeapSelector(k=0)

    with pytest.raises(ValueError, match="must be a positive integer"):
        TopKHeapSelector(k=-5)


def test_empty_selector() -> None:
    """Verifies behavior when no items are pushed (M = 0)."""
    selector = TopKHeapSelector(k=5)
    assert len(selector) == 0
    assert selector.capacity == 5
    assert selector.get_top_k() == []
    assert selector.get_top_k_ids() == []


def test_fewer_items_than_capacity() -> None:
    """Verifies behavior when incoming candidates M < capacity N."""
    selector = TopKHeapSelector(k=5)
    items = [(101, 2.5), (102, 7.1), (103, 1.0)]
    selector.update(items)

    assert len(selector) == 3
    expected = [(102, 7.1), (101, 2.5), (103, 1.0)]
    assert selector.get_top_k() == expected
    assert selector.get_top_k_ids() == [102, 101, 103]


def test_capacity_truncation() -> None:
    """Verifies that only top-K elements are retained when M > N."""
    selector = TopKHeapSelector(k=3)
    items = [
        (1, 10.0),
        (2, 5.0),
        (3, 20.0),
        (4, 1.0),
        (5, 15.0),
    ]
    selector.update(items)

    assert len(selector) == 3
    # Top 3: 3 (20.0), 5 (15.0), 1 (10.0)
    assert selector.get_top_k() == [(3, 20.0), (5, 15.0), (1, 10.0)]


def test_negative_scores() -> None:
    """Verifies correct ranking with negative and zero weights."""
    selector = TopKHeapSelector(k=3)
    items = [
        (1, -100.5),
        (2, -10.0),
        (3, 0.0),
        (4, -5.2),
        (5, -50.0),
    ]
    selector.update(items)

    expected = [(3, 0.0), (4, -5.2), (2, -10.0)]
    assert selector.get_top_k() == expected


def test_nan_score_raises_value_error() -> None:
    """Verifies that pushing float('nan') raises ValueError to protect heap invariant."""
    selector = TopKHeapSelector(k=3)
    with pytest.raises(ValueError, match="Score cannot be NaN"):
        selector.push(item_id=1, score=float("nan"))


def test_deterministic_tie_breaking() -> None:
    """Verifies that on equal scores, lower item_id takes precedence."""
    selector = TopKHeapSelector(k=2)
    # Все скоры равны 10.0
    items = [
        (50, 10.0),
        (10, 10.0),
        (30, 10.0),
        (20, 10.0),
    ]
    selector.update(items)

    # При k=2 должны остаться элементы с наименьшими ID: 10 и 20
    assert selector.get_top_k() == [(10, 10.0), (20, 10.0)]
    assert selector.get_top_k_ids() == [10, 20]


def test_clear_method() -> None:
    """Verifies that clear resets the heap state."""
    selector = TopKHeapSelector(k=2)
    selector.push(1, 5.0)
    selector.push(2, 6.0)
    assert len(selector) == 2

    selector.clear()
    assert len(selector) == 0
    assert selector.get_top_k() == []


def test_heap_entry_equality_and_comparison() -> None:
    """Unit test for HeapEntry equality and edge comparison branches."""
    e1 = HeapEntry(score=5.0, item_id=10)
    e2 = HeapEntry(score=5.0, item_id=10)
    e3 = HeapEntry(score=5.0, item_id=20)
    e4 = HeapEntry(score=6.0, item_id=5)

    assert e1 == e2
    assert e1 != e3
    assert (e1 == "not_an_entry") is False
    # e3 has larger id -> in min-heap e3 < e1 (e3 is worse, evicted earlier)
    assert e3 < e1
    # e4 has larger score -> e1 < e4
    assert e1 < e4


def test_equivalence_with_naive_sort() -> None:
    """Property-based stress test: results must strictly match naive sorting."""
    rng = random.Random(42)
    candidates = [(rng.randint(1, 200), rng.uniform(-50.0, 50.0)) for _ in range(1_000)]
    k = 15

    selector = TopKHeapSelector(k=k)
    selector.update(candidates)
    heap_result = selector.get_top_k()

    # Наивная сортировка всего массива: -score, затем item_id
    naive_sorted = sorted(candidates, key=lambda x: (-x[1], x[0]))[:k]

    assert heap_result == naive_sorted
