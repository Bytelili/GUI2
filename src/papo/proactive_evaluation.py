from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data_protocol import episode_keys, sha256_file
from .io import write_jsonl
from .official_data import read_csv_rows
from .tasks import build_proactive_suggestion_tasks


def prepare_proactive_evaluation_tasks(
    *,
    official_root: str | Path,
    protocol_dir: str | Path,
    raw_root: str | Path,
    output_dir: str | Path,
    screenshot_level: int,
    history_limit: int,
    require_complete: bool,
    test_split: str,
) -> dict[str, Any]:
    official_root = Path(official_root)
    protocol_dir = Path(protocol_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_path = official_root / test_split
    strict_history_path = protocol_dir / "proactive_history.csv"
    official_history_path = official_root / "total.csv"
    profiles_path = official_root / "user_profile.csv"

    modes = {
        "strict_holdout": strict_history_path,
        "official_online": official_history_path,
    }
    report: dict[str, Any] = {
        "status": "passed",
        "test_split": test_split,
        "test_split_sha256": sha256_file(test_path),
        "screenshot_level": screenshot_level,
        "history_limit": history_limit,
        "modes": {},
    }
    test_ids = _episode_ids(read_csv_rows(test_path))
    strict_ids = _episode_ids(read_csv_rows(strict_history_path))

    for mode, history_path in modes.items():
        tasks = build_proactive_suggestion_tasks(
            test_path,
            history_path,
            profiles_path,
            raw_root,
            screenshot_level=screenshot_level,
            history_limit=history_limit,
            require_complete=require_complete,
            provenance={
                "partition": "official_test",
                "evaluation_history_mode": mode,
                "target_split": test_split,
                "history_split": history_path.name,
            },
        )
        audit = validate_proactive_evaluation_tasks(tasks, test_ids, strict_ids, mode)
        task_path = output_dir / f"proactive_test_{mode}_level_{screenshot_level}.jsonl"
        write_jsonl(task_path, tasks)
        report["modes"][mode] = {
            **audit,
            "task_path": str(task_path),
            "task_sha256": sha256_file(task_path),
            "history_path": str(history_path),
            "history_sha256": sha256_file(history_path),
        }

    report_path = output_dir / f"proactive_test_task_audit_level_{screenshot_level}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def validate_proactive_evaluation_tasks(
    tasks: list[dict[str, Any]],
    test_ids: set[str],
    strict_history_ids: set[str],
    mode: str,
) -> dict[str, Any]:
    target_ids: set[str] = set()
    history_ids: set[str] = set()
    duplicate_targets = 0
    non_temporal = 0
    self_history = 0
    for task in tasks:
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
        target = task.get("target") if isinstance(task.get("target"), dict) else {}
        target_id = str(metadata.get("papo_episode_id") or "")
        target_time = str(inputs.get("time") or "")
        if not target_id or not target_time or not str(target.get("intent") or ""):
            raise ValueError(f"{mode} evaluation contains an incomplete target task")
        duplicate_targets += target_id in target_ids
        target_ids.add(target_id)
        if str(metadata.get("evaluation_history_mode") or "") != mode:
            raise ValueError(f"{mode} evaluation contains a mismatched history-mode label")
        for item in inputs.get("previous_intents", []):
            history_id = str(item.get("episode_id") or "")
            history_ids.add(history_id)
            non_temporal += str(item.get("time") or "") >= target_time
            self_history += history_id == target_id

    if not target_ids <= test_ids:
        raise ValueError(f"{mode} evaluation contains targets outside the official test split")
    if duplicate_targets:
        raise ValueError(f"{mode} evaluation contains {duplicate_targets} duplicate targets")
    if non_temporal or self_history:
        raise ValueError(
            f"{mode} evaluation violates temporal causality: non_temporal={non_temporal}, self_history={self_history}"
        )
    outside_strict = history_ids - strict_history_ids
    test_history = history_ids & test_ids
    if mode == "strict_holdout" and (outside_strict or test_history):
        raise ValueError(
            f"strict_holdout history leakage: outside_strict={len(outside_strict)}, test_history={len(test_history)}"
        )
    if mode not in {"strict_holdout", "official_online"}:
        raise ValueError(f"Unknown proactive evaluation history mode: {mode}")
    return {
        "official_test_targets": len(test_ids),
        "tasks": len(tasks),
        "unique_targets": len(target_ids),
        "excluded_incomplete_targets": len(test_ids - target_ids),
        "unique_history_episodes": len(history_ids),
        "history_episodes_outside_strict_train": len(outside_strict),
        "history_episodes_from_test_suggestion": len(test_history),
        "temporal_violations": non_temporal,
        "self_history_violations": self_history,
    }


def _episode_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {f"{user_id}__{time}" for user_id, time in episode_keys(rows)}
