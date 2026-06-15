from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from .io_utils import stable_id


SPACE_PATTERN = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    return SPACE_PATTERN.sub("", str(value or "").strip()).lower()


def build_candidate_sets(
    target_tasks: list[dict[str, Any]],
    *,
    reference_tasks: list[dict[str, Any]],
    partition: str,
    model_candidates: dict[str, list[str]] | None = None,
    max_same_user: int = 2,
    max_cross_user: int = 3,
    max_model: int = 3,
) -> list[dict[str, Any]]:
    """Build deterministic candidate sets without using eval targets as references."""
    if partition not in {"train", "eval"}:
        raise ValueError(f"Unknown partition: {partition}")
    model_candidates = model_candidates or {}
    reference_records = [_task_record(task) for task in reference_tasks]
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in reference_records:
        by_user[record["user_id"]].append(record)

    results: list[dict[str, Any]] = []
    for task in target_tasks:
        inputs = _inputs(task)
        target = _target(task)
        metadata = _metadata(task)
        task_id = str(task.get("task_id") or "")
        user_id = str(inputs.get("user_id") or "")
        target_time = str(inputs.get("time") or "")
        target_text = str(target.get("intent") or "").strip()
        if not task_id or not target_text:
            raise ValueError("Preference target task is missing task_id or target intent")

        candidates: list[dict[str, Any]] = [
            _candidate(
                task_id,
                target_text,
                "oracle_target",
                source_episode_id=str(metadata.get("papo_episode_id") or ""),
                source_user_id=user_id,
                source_time=target_time,
                source_scenario=str(inputs.get("scenario") or ""),
            )
        ]

        history = [
            {
                "intent": str(item.get("intent") or "").strip(),
                "episode_id": str(item.get("episode_id") or ""),
                "user_id": user_id,
                "time": str(item.get("time") or ""),
                "scenario": str(item.get("scenario") or ""),
            }
            for item in inputs.get("previous_intents", [])
            if str(item.get("intent") or "").strip()
        ]
        history.sort(
            key=lambda item: (
                _context_score(inputs, item),
                item["time"],
            ),
            reverse=True,
        )
        for item in history[:max_same_user]:
            candidates.append(
                _candidate(
                    task_id,
                    item["intent"],
                    "same_user_history",
                    source_episode_id=item["episode_id"],
                    source_user_id=user_id,
                    source_time=item["time"],
                    source_scenario=item["scenario"],
                )
            )

        cross = [
            record
            for record in reference_records
            if record["user_id"] != user_id and (not target_time or record["time"] < target_time)
        ]
        cross.sort(
            key=lambda item: (
                _context_score(inputs, item),
                _text_similarity(target_text, item["intent"]),
                item["time"],
            ),
            reverse=True,
        )
        for item in cross[:max_cross_user]:
            candidates.append(
                _candidate(
                    task_id,
                    item["intent"],
                    "cross_user_hard",
                    source_episode_id=item["episode_id"],
                    source_user_id=item["user_id"],
                    source_time=item["time"],
                    source_scenario=item["scenario"],
                )
            )

        for text in model_candidates.get(task_id, [])[:max_model]:
            candidates.append(_candidate(task_id, text, "sft_sample"))

        candidates = _deduplicate(candidates)
        if len(candidates) < 2:
            continue
        results.append(
            {
                "task_id": task_id,
                "partition": partition,
                "input": inputs,
                "target": target,
                "metadata": metadata,
                "candidate_reference_partition": "train",
                "candidates": candidates,
            }
        )
    return results


def model_candidate_map(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        candidates = [
            str(item).strip()
            for item in row.get("candidates", [])
            if str(item).strip()
        ]
        if task_id and candidates:
            result[task_id] = candidates
    return result


def _candidate(
    task_id: str,
    text: str,
    source: str,
    *,
    source_episode_id: str = "",
    source_user_id: str = "",
    source_time: str = "",
    source_scenario: str = "",
) -> dict[str, Any]:
    clean = str(text or "").strip()
    return {
        "candidate_id": stable_id(task_id, source, normalize_text(clean), source_episode_id),
        "text": clean,
        "source": source,
        "source_episode_id": source_episode_id,
        "source_user_id": source_user_id,
        "source_time": source_time,
        "source_scenario": source_scenario,
    }


def _deduplicate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        key = normalize_text(candidate.get("text"))
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _task_record(task: dict[str, Any]) -> dict[str, Any]:
    inputs = _inputs(task)
    target = _target(task)
    metadata = _metadata(task)
    return {
        "episode_id": str(metadata.get("papo_episode_id") or ""),
        "user_id": str(inputs.get("user_id") or ""),
        "time": str(inputs.get("time") or ""),
        "scenario": str(inputs.get("scenario") or ""),
        "intent": str(target.get("intent") or ""),
    }


def _context_score(inputs: dict[str, Any], candidate: dict[str, Any]) -> float:
    score = float(str(inputs.get("scenario") or "") == str(candidate.get("scenario") or ""))
    target_hour = _hour(str(inputs.get("time") or ""))
    candidate_hour = _hour(str(candidate.get("time") or ""))
    if target_hour is not None and candidate_hour is not None:
        distance = min(abs(target_hour - candidate_hour), 24 - abs(target_hour - candidate_hour))
        score += 1.0 - distance / 12.0
    return score


def _hour(timestamp: str) -> int | None:
    try:
        return int(timestamp.split("_", 1)[1][:2])
    except (IndexError, TypeError, ValueError):
        return None


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def _inputs(task: dict[str, Any]) -> dict[str, Any]:
    return dict(task.get("input")) if isinstance(task.get("input"), dict) else {}


def _target(task: dict[str, Any]) -> dict[str, Any]:
    return dict(task.get("target")) if isinstance(task.get("target"), dict) else {}


def _metadata(task: dict[str, Any]) -> dict[str, Any]:
    return dict(task.get("metadata")) if isinstance(task.get("metadata"), dict) else {}
