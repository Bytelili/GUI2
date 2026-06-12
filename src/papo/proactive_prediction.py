from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .data_protocol import sha256_file
from .llamafactory_export import SYSTEM_PROMPT, proactive_prompt


RESULT_FIELDS = [
    "task_id",
    "episode_id",
    "user_id",
    "episode_time",
    "history_mode",
    "screenshot_level",
    "original_intent",
    "predicted_intent",
    "time",
    "token",
    "prompt_token",
    "response_token",
    "finish_reason",
    "error",
]


def build_inference_request(task: dict[str, Any]) -> dict[str, Any]:
    inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
    return {
        "messages": [{"role": "user", "content": proactive_prompt(inputs)}],
        "system": SYSTEM_PROMPT,
        "images": [str(path) for path in inputs.get("initial_screenshots", []) if str(path)],
    }


def prediction_record(
    task: dict[str, Any],
    *,
    predicted_intent: str,
    elapsed_seconds: float,
    prompt_tokens: int,
    response_tokens: int,
    finish_reason: str,
    error: str = "",
) -> dict[str, Any]:
    inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
    target = task.get("target") if isinstance(task.get("target"), dict) else {}
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    return {
        "task_id": str(task.get("task_id") or ""),
        "episode_id": str(metadata.get("papo_episode_id") or ""),
        "user_id": str(inputs.get("user_id") or ""),
        "episode_time": str(inputs.get("time") or ""),
        "history_mode": str(metadata.get("evaluation_history_mode") or ""),
        "screenshot_level": int(metadata.get("screenshot_level", 0) or 0),
        "original_intent": str(target.get("intent") or ""),
        "predicted_intent": str(predicted_intent or "").strip() or "ERROR",
        "time": round(float(elapsed_seconds), 4),
        "token": int(prompt_tokens) + int(response_tokens),
        "prompt_token": int(prompt_tokens),
        "response_token": int(response_tokens),
        "finish_reason": str(finish_reason or ""),
        "error": str(error or ""),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def prepare_prediction_resume(
    tasks: list[dict[str, Any]],
    output_path: str | Path,
) -> tuple[set[str], int]:
    assigned_ids = [str(task.get("task_id") or "") for task in tasks]
    if any(not task_id for task_id in assigned_ids) or len(set(assigned_ids)) != len(assigned_ids):
        raise ValueError("Assigned prediction tasks contain empty or duplicate task IDs")

    output = Path(output_path)
    rows, truncated_removed = _read_resumable_jsonl(output)
    seen: set[str] = set()
    successful: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if task_id not in assigned_ids:
            raise ValueError(f"Prediction resume contains an unassigned task ID: {task_id}")
        if task_id in seen:
            raise ValueError(f"Prediction resume contains a duplicate task ID: {task_id}")
        seen.add(task_id)
        if _successful_prediction(row):
            successful.append(row)

    failed_removed = len(rows) - len(successful) + truncated_removed
    if failed_removed:
        _write_jsonl_atomic(output, successful)
    return {str(row.get("task_id") or "") for row in successful}, failed_removed


def merge_prediction_shards(
    tasks: list[dict[str, Any]],
    shard_paths: Iterable[str | Path],
    output_csv: str | Path,
    *,
    task_path: str | Path,
    adapter_dir: str | Path,
    allow_errors: bool = False,
) -> dict[str, Any]:
    task_ids = [str(task.get("task_id") or "") for task in tasks]
    if any(not task_id for task_id in task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("Prediction task file contains empty or duplicate task IDs")
    task_by_id = {str(task.get("task_id") or ""): task for task in tasks}
    records: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for shard_path in shard_paths:
        for row in read_jsonl(shard_path):
            task_id = str(row.get("task_id") or "")
            if task_id in records:
                duplicates.add(task_id)
            records[task_id] = row

    unknown = set(records) - set(task_by_id)
    missing = set(task_by_id) - set(records)
    errors = [task_id for task_id, row in records.items() if str(row.get("error") or "") or row.get("predicted_intent") == "ERROR"]
    if duplicates or unknown or missing or (errors and not allow_errors):
        raise ValueError(
            "Prediction merge validation failed: "
            f"duplicates={len(duplicates)}, unknown={len(unknown)}, missing={len(missing)}, errors={len(errors)}"
        )

    ordered = [records[str(task.get("task_id") or "")] for task in tasks]
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in RESULT_FIELDS} for row in ordered)

    adapter_dir = Path(adapter_dir)
    provenance_path = adapter_dir / "papo_training_provenance.json"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    report = {
        "status": "passed",
        "task_path": str(Path(task_path)),
        "task_sha256": sha256_file(task_path),
        "adapter_dir": str(adapter_dir),
        "adapter_sha256": sha256_file(adapter_path),
        "adapter_provenance_sha256": sha256_file(provenance_path),
        "output_csv": str(output),
        "output_sha256": sha256_file(output),
        "records": len(ordered),
        "errors": len(errors),
        "history_modes": sorted({str(row.get("history_mode") or "") for row in ordered}),
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _successful_prediction(row: dict[str, Any]) -> bool:
    return not str(row.get("error") or "") and str(row.get("predicted_intent") or "").strip().upper() != "ERROR"


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
    temporary.replace(path)


def _read_resumable_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], 0
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError(f"Prediction resume contains a corrupt non-final JSONL row: {path}") from None
            return rows, 1
    return rows, 0
