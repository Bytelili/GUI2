from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def action_label(step: dict[str, Any]) -> str:
    direct = str(step.get("action") or "").strip()
    if direct:
        return direct

    metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
    label = str(metadata.get("action_frame_label") or "").strip()
    if label:
        return label

    frame = step.get("action_frame") if isinstance(step.get("action_frame"), dict) else {}
    label = str(frame.get("frame_label") or "").strip()
    if label:
        return label

    normalized = step.get("normalized_action")
    if isinstance(normalized, dict):
        action_type = str(normalized.get("action_type") or "unknown")
        if action_type == "scroll":
            return f"scroll:{normalized.get('direction') or 'unknown'}"
        if action_type == "type":
            return "input:TextField"
        return action_type

    return "unknown"


def state_key(step: dict[str, Any]) -> str:
    direct = str(step.get("state_key") or "").strip()
    if direct:
        return direct
    parts = [
        str(step.get("app") or ""),
        str(step.get("stage_label") or ""),
        str(step.get("ui_signature") or ""),
    ]
    return "|".join(parts)


def intent_key(step: dict[str, Any]) -> str:
    return str(step.get("intent_key") or step.get("intent_signature") or step.get("intent") or "")


def step_id(step: dict[str, Any]) -> str:
    return str(step.get("papo_step_id") or step.get("step_id") or step.get("raw_step_id") or "")
