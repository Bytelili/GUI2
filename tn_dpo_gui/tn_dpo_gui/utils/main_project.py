from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import yaml

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def find_repo_root(start: Path | None = None) -> Path:
    cursor = (start or Path(__file__).resolve()).resolve()
    for parent in [cursor, *cursor.parents]:
        if (parent / "config.yaml").exists() and (parent / "src" / "papo" / "config.py").exists():
            return parent
    raise FileNotFoundError("Could not locate the main project root containing config.yaml and src/papo/config.py")


def load_main_project_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path).resolve() if config_path else find_repo_root() / "config.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    expanded = _expand(payload)
    expanded["_config_path"] = str(path)
    expanded["_project_root"] = str(path.parent)
    return expanded


def config_value(config: dict[str, Any], key: str) -> Any:
    value: Any = config
    for token in key.split("."):
        value = value[token]
    return value


def config_path(config: dict[str, Any], key: str) -> Path:
    return _resolve_path_like(config_value(config, key), Path(config["_project_root"]))


def model_location(config: dict[str, Any]) -> str:
    value = str(config.get("training", {}).get("model_name_or_path") or config.get("paths", {}).get("qwen_model_path") or "")
    if _looks_absolute(value) or value.startswith("."):
        return str(_resolve_path_like(value, Path(config["_project_root"])))
    return value


def stable_task_id(instruction: str) -> str:
    return "intent::" + hashlib.md5((instruction or "").strip().encode("utf-8")).hexdigest()[:12]


def derive_tn_dpo_layout(root_config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_main_project_config(root_config_path)
    work_dir = config_path(config, "paths.work_dir")
    task_dir = config_path(config, "paths.task_dir")
    checkpoint_root = config_path(config, "paths.checkpoint_root")
    logging_root = config_path(config, "paths.logging_root")
    integration_dir = work_dir / "tn_dpo_gui"
    processed_dir = integration_dir / "processed"
    pairs_dir = integration_dir / "pairs"
    checkpoint_dir = checkpoint_root / "tn_dpo_gui"
    logging_dir = logging_root / "tn_dpo_gui"
    return {
        "repo_root": Path(config["_project_root"]),
        "root_config_path": Path(config["_config_path"]),
        "raw_root": config_path(config, "paths.raw_root"),
        "official_root": config_path(config, "paths.official_root"),
        "protocol_dir": config_path(config, "paths.protocol_dir"),
        "work_dir": work_dir,
        "task_dir": task_dir,
        "llamafactory_dir": config_path(config, "paths.llamafactory_dir"),
        "llamafactory_data_dir": config_path(config, "paths.llamafactory_data_dir"),
        "checkpoint_root": checkpoint_root,
        "logging_root": logging_root,
        "model_name_or_path": model_location(config),
        "integration_dir": integration_dir,
        "processed_dir": processed_dir,
        "steps_path": processed_dir / "steps.jsonl",
        "trajectories_path": processed_dir / "trajectories.jsonl",
        "user_index_path": processed_dir / "user_index.json",
        "preprocess_summary_path": processed_dir / "summary.json",
        "pairs_path": pairs_dir / "pairs.jsonl",
        "pair_summary_path": pairs_dir / "summary.json",
        "ranker_dir": checkpoint_dir / "ranker",
        "gate_dir": checkpoint_dir / "gate",
        "ranker_path": checkpoint_dir / "ranker" / "ranker.pt",
        "gate_path": checkpoint_dir / "gate" / "gate.pt",
        "eval_report_path": logging_dir / "report.json",
        "train_tasks_path": task_dir / "execution_train_config.jsonl",
        "eval_tasks_path": task_dir / "execution_eval_config.jsonl",
        "papo_steps_path": work_dir / "papo_steps.jsonl",
    }


def validate_integration_inputs(layout: dict[str, Any]) -> None:
    required = [
        layout["train_tasks_path"],
        layout["eval_tasks_path"],
        layout["papo_steps_path"],
    ]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing main-project TN-DPO inputs. Run the main PAPO preparation pipeline first. Missing:\n"
            + "\n".join(missing)
        )


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), match.group(2) or ""), value)
    return value


def _looks_absolute(value: str) -> bool:
    return bool(value) and (Path(value).is_absolute() or value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", value) is not None)


def _resolve_path_like(value: Any, project_root: Path) -> Path:
    text = str(value)
    if text.startswith("~"):
        return Path(text).expanduser()
    if _looks_absolute(text):
        return Path(text)
    return project_root / Path(text)
