from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .io import read_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    return read_yaml(path)


def merge_overrides(base: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = deepcopy(base)
    if not overrides:
        return merged
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_overrides(merged[key], value)
        else:
            merged[key] = value
    return merged
