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


def _config_base_dir(config: dict | None = None) -> Path:
    config_dir = (config or {}).get("_config_dir")
    return Path(config_dir) if config_dir else PROJECT_ROOT


def resolve_config_path_value(path_like: str, base_dir: Path | None = None) -> str:
    text = str(path_like)
    if not text:
        return text
    anchor = base_dir or PROJECT_ROOT
    if text.startswith("~"):
        return path_to_string(Path(text).expanduser())
    if _looks_absolute_path_string(text):
        if text.startswith("/") and not text.startswith("//"):
            return text.replace("\\", "/")
        return path_to_string(Path(text))
    return path_to_string((anchor / Path(text)).resolve())


def resolve_config_paths(config: dict, mapping: dict[str, list[str]]) -> dict:
    base_dir = _config_base_dir(config)
    for section, keys in mapping.items():
        if section not in config:
            continue
        for key in keys:
            if key in config[section]:
                config[section][key] = resolve_config_path_value(config[section][key], base_dir=base_dir)
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
    base_dir = _config_base_dir(resolved)
    layout = derive_tn_dpo_layout(resolve_config_path_value(root_config, base_dir=base_dir) if root_config else None)
    resolved["_main_project_layout"] = {key: path_to_string(value) if isinstance(value, Path) else value for key, value in layout.items()}
    for section, values in mapping.items():
        resolved.setdefault(section, {})
        for key, layout_key in values.items():
            current = resolved[section].get(key)
            if current in (None, "", "auto"):
                value = layout[layout_key]
                resolved[section][key] = path_to_string(value) if isinstance(value, Path) else value
    return resolved
