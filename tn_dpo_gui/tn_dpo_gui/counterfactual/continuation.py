from __future__ import annotations

from collections.abc import Callable, Iterable

from tn_dpo_gui.data.schema import TrajectoryContinuation


def select_min_distance_pair(
    left_candidates: Iterable[TrajectoryContinuation],
    right_candidates: Iterable[TrajectoryContinuation],
    distance_fn: Callable[[TrajectoryContinuation, TrajectoryContinuation], float],
) -> tuple[TrajectoryContinuation | None, TrajectoryContinuation | None, float]:
    best_left = None
    best_right = None
    best_distance = float("inf")
    for left in left_candidates:
        for right in right_candidates:
            distance = float(distance_fn(left, right))
            if distance < best_distance:
                best_left = left
                best_right = right
                best_distance = distance
    return best_left, best_right, best_distance
