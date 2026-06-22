from __future__ import annotations

from typing import Iterable, TypeVar


T = TypeVar("T")
HISTORICAL_SPLITS = {"train", "history"}


def is_history_split(split: str | None) -> bool:
    return (split or "").lower() in HISTORICAL_SPLITS


def assert_no_test_splits(splits: Iterable[str]) -> None:
    forbidden = [split for split in splits if (split or "").lower() == "test"]
    if forbidden:
        raise ValueError("History retrieval cannot read from test split data.")


def filter_by_split(items: Iterable[T], allowed_splits: set[str], split_attr: str = "split") -> list[T]:
    normalized = {split.lower() for split in allowed_splits}
    return [item for item in items if getattr(item, split_attr, "").lower() in normalized]
