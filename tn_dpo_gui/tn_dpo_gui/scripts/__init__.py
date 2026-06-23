from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from tn_dpo_gui.utils.main_project import derive_tn_dpo_layout, path_to_string

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _looks_absolute_path_string(text: str) -> bool:
    return bool(text) and (Path(text).is_absolute() or text.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", text))


def resolve_path(path_like: str) -> Path:
    text = str(path_like)
    path = Path(text)
    if _looks_absolute_path_string(text):
        return path
    return PROJECT_ROOT / path


def resolve_config_path_value(path_like: str) -> str:
    text = str(path_like)
    if not text:
        return text
    if text.startswith("~"):
        return path_to_string(Path(text).expanduser())
    if _looks_absolute_path_string(text):
        if text.startswith("/") and not text.startswith("//"):
            return text.replace("\\", "/")
        return path_to_string(Path(text))
    return path_to_string(PROJECT_ROOT / Path(text))


def resolve_config_paths(config: dict, mapping: dict[str, list[str]]) -> dict:
    for section, keys in mapping.items():
        if section not in config:
            continue
        for key in keys:
            if key in config[section]:
                config[section][key] = resolve_config_path_value(config[section][key])
    return config


def override_main_project_root_config(config: dict, root_config: str | None) -> dict:
    if root_config:
        config.setdefault("main_project", {})["root_config"] = root_config
    return config


def apply_main_project_layout(config: dict, mapping: dict[str, dict[str, str]]) -> dict:
    resolved = deepcopy(config)
    integration = resolved.get("main_project") or {}
    if not integration.get("enabled", False):
        return resolved

    root_config = integration.get("root_config")
    layout = derive_tn_dpo_layout(resolve_path(root_config) if root_config else None)
    resolved["_main_project_layout"] = {key: path_to_string(value) if isinstance(value, Path) else value for key, value in layout.items()}
    for section, values in mapping.items():
        resolved.setdefault(section, {})
        for key, layout_key in values.items():
            current = resolved[section].get(key)
            if current in (None, "", "auto"):
                value = layout[layout_key]
                resolved[section][key] = path_to_string(value) if isinstance(value, Path) else value
    return resolved
