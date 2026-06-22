from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_config_paths(config: dict, mapping: dict[str, list[str]]) -> dict:
    for section, keys in mapping.items():
        if section not in config:
            continue
        for key in keys:
            if key in config[section]:
                config[section][key] = str(resolve_path(config[section][key]))
    return config
