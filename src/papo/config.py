from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    expanded = _expand(data)
    expanded["_config_path"] = str(config_path)
    expanded["_project_root"] = str(config_path.parent)
    return expanded


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def config_path(config: dict[str, Any], key: str) -> Path:
    value: Any = config
    for token in key.split("."):
        value = value[token]
    return resolve_path(config, value)


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), match.group(2) or ""), value)
    return value
