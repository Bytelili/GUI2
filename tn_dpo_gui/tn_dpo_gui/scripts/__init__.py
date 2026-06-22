from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from tn_dpo_gui.utils.main_project import derive_tn_dpo_layout

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


def apply_main_project_layout(config: dict, mapping: dict[str, dict[str, str]]) -> dict:
    resolved = deepcopy(config)
    integration = resolved.get("main_project") or {}
    if not integration.get("enabled", False):
        return resolved

    root_config = integration.get("root_config")
    layout = derive_tn_dpo_layout(resolve_path(root_config) if root_config else None)
    resolved["_main_project_layout"] = {key: str(value) if isinstance(value, Path) else value for key, value in layout.items()}
    for section, values in mapping.items():
        resolved.setdefault(section, {})
        for key, layout_key in values.items():
            current = resolved[section].get(key)
            if current in (None, "", "auto"):
                value = layout[layout_key]
                resolved[section][key] = str(value) if isinstance(value, Path) else value
    return resolved
