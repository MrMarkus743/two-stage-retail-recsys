"""High-performance Top-K selector based on a Binary Min-Heap.

Ensures O(M log N) time complexity and O(N) auxiliary space for selecting
the top N items with deterministic tie-breaking.
"""

import heapq
import math
from collections.abc import Iterable
from functools import total_ordering


@total_ordering
class HeapEntry:
    """Internal heap entry wrapper providing deterministic priority comparisons.

    Min-heap ordering criteria:
    - Lower score is evicted first (self.score < other.score).
    - On equal scores, higher item_id is evicted first (self.item_id > other.item_id),
      ensuring that lower item_ids have higher priority to remain in the Top-K.
    """

    __slots__ = ("item_id", "score")

    def __init__(self, score: float, item_id: int) -> None:
        self.score = float(score)
        self.item_id = int(item_id)

    def __lt__(self, other: "HeapEntry") -> bool:
        if self.score != other.score:
            return self.score < other.score
        return self.item_id > other.item_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HeapEntry):
            return NotImplemented
        return self.score == other.score and self.item_id == other.item_id


class TopKHeapSelector:
    """Fixed-capacity Binary Min-Heap Top-K selector."""

    def __init__(self, k: int) -> None:
        """Initializes the heap selector with capacity k.

        Args:
            k: Maximum number of elements to maintain in the Top-K set (k > 0).

        Raises:
            ValueError: If k <= 0.
        """
        if k <= 0:
            raise ValueError(f"Capacity k must be a positive integer, got {k}")
        self._k = k
        self._heap: list[HeapEntry] = []

    @property
    def capacity(self) -> int:
        """Returns the maximum capacity k."""
        return self._k

    def __len__(self) -> int:
        """Returns the current number of elements in the selector."""
        return len(self._heap)

    def push(self, item_id: int, score: float) -> None:
        """Pushes an (item_id, score) candidate into the selector.

        Args:
            item_id: Integer identifier of the item.
            score: Numeric relevance score.

        Raises:
            ValueError: If score is NaN.
        """
        if math.isnan(score):
            raise ValueError(f"Score cannot be NaN (item_id: {item_id})")

        entry = HeapEntry(score=score, item_id=item_id)

        if len(self._heap) < self._k:
            heapq.heappush(self._heap, entry)
        elif entry > self._heap[0]:
            heapq.heapreplace(self._heap, entry)

    def update(self, items: Iterable[tuple[int, float]]) -> None:
        """Batch updates the selector from an iterable of (item_id, score) pairs.

        Args:
            items: Iterable yielding (item_id, score) tuples.
        """
        for item_id, score in items:
            self.push(item_id, score)

    def clear(self) -> None:
        """Clears all elements from the selector."""
        self._heap.clear()

    def get_top_k(self) -> list[tuple[int, float]]:
        """Extracts top elements sorted by score descending (tie-broken by item_id ascending).

        Does not mutate the internal heap state.

        Returns:
            List of (item_id, score) tuples in descending order of relevance.
        """
        sorted_entries = sorted(
            self._heap,
            key=lambda e: (-e.score, e.item_id),
        )
        return [(e.item_id, e.score) for e in sorted_entries]

    def get_top_k_ids(self) -> list[int]:
        """Extracts only the item_ids of the top elements in descending order.

        Returns:
            List of item_ids in descending order of relevance.
        """
        return [item_id for item_id, _ in self.get_top_k()]
