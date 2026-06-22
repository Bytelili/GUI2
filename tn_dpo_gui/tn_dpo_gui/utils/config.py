from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from .io import read_yaml

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_config(path: str | Path) -> dict[str, Any]:
    return _expand(read_yaml(path))


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


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), match.group(2) or ""), value)
    return value
