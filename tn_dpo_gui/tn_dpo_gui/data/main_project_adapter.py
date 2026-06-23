from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any

from tn_dpo_gui.utils.io import read_jsonl
from tn_dpo_gui.utils.main_project import stable_task_id

from .action_schema import Action
from .schema import GUIStepExample, TrajectoryRecord


def convert_main_project_artifacts(
    train_tasks_path: str | Path,
    eval_tasks_path: str | Path,
    steps_path: str | Path,
) -> tuple[list[GUIStepExample], list[TrajectoryRecord], dict[str, Any]]:
    train_tasks = read_jsonl(train_tasks_path)
    eval_tasks = read_jsonl(eval_tasks_path)
    steps = read_jsonl(steps_path)
    steps_by_episode = _group_steps_by_episode(steps)
    _validate_target_episode_coverage(train_tasks, eval_tasks, steps_by_episode)

    examples = _build_examples(train_tasks, steps_by_episode, split="train")
    examples.extend(_build_examples(eval_tasks, steps_by_episode, split="eval"))

    trajectories = _build_train_trajectories(train_tasks, steps_by_episode)
    trajectories.extend(_build_history_trajectories(train_tasks + eval_tasks, steps_by_episode, existing_ids={row.trajectory_id for row in trajectories}))

    summary = {
        "train_tasks": len(train_tasks),
        "eval_tasks": len(eval_tasks),
        "papo_steps": len(steps),
        "examples": len(examples),
        "trajectories": len(trajectories),
        "train_examples": sum(example.split == "train" for example in examples),
        "eval_examples": sum(example.split == "eval" for example in examples),
        "history_trajectories": sum(record.split == "history" for record in trajectories),
    }
    return examples, trajectories, summary


def parse_action_label(
    label: str,
    raw_action: str | None = None,
    object_text: str | None = None,
    object_role: str | None = None,
) -> Action:
    normalized = str(label or "").strip()
    raw = {"label": normalized}
    if raw_action:
        raw["raw_action"] = str(raw_action)
    if object_text:
        raw["object_text"] = str(object_text)
    if object_role:
        raw["object_role"] = str(object_role)
    if not normalized:
        return Action(action_type="unknown", raw=raw)
    if normalized in {"finished", "finish"}:
        return Action(action_type="finish", target="terminal", raw=raw)
    if normalized == "wait":
        return Action(action_type="wait", target="pause", raw=raw)
    if normalized.startswith("scroll:"):
        return Action(action_type="scroll", text=normalized.split(":", 1)[1], raw=raw)
    if normalized.startswith("navigate:"):
        return Action(action_type="click", target=normalized.split(":", 1)[1], raw=raw)
    if normalized.startswith("input:"):
        target = normalized.split(":", 1)[1].strip() or str(object_role or "").strip() or "TextField"
        text = _extract_input_text(raw_action)
        return Action(action_type="type", target=target, text=text or None, raw=raw)
    if ":" in normalized:
        verb, target = normalized.split(":", 1)
        target = target.strip() or str(object_text or "").strip() or str(object_role or "").strip()
        action_type = {
            "select": "click",
            "submit": "click",
            "focus": "click",
            "open": "click",
            "press": "hotkey",
        }.get(verb.strip(), verb.strip())
        if action_type == "hotkey":
            return Action(action_type="hotkey", text=target or verb, raw=raw)
        return Action(action_type=action_type, target=target or verb, raw=raw)
    return Action(action_type="freeform", text=normalized, raw=raw)


def _build_examples(tasks: list[dict[str, Any]], steps_by_episode: dict[str, list[dict[str, Any]]], split: str) -> list[GUIStepExample]:
    examples: list[GUIStepExample] = []
    for task in tasks:
        episode_id = _episode_id_from_task(task)
        episode_steps = steps_by_episode.get(episode_id, [])
        instruction = _task_instruction(task)
        task_id = stable_task_id(instruction)
        goal_state = _goal_state(task, episode_steps)

        if not episode_steps:
            continue

        num_steps = len(episode_steps)
        trajectory_id = f"episode::{episode_id}"
        episode_success = 1.0 if bool(episode_steps[-1].get("is_terminal")) else 0.0
        for index, step in enumerate(episode_steps):
            action_history = [_parse_step_action(prev) for prev in episode_steps[:index]]
            current_action = _parse_step_action(step)
            future_trajectory = [_parse_step_action(row) for row in episode_steps[index + 1 :]]
            examples.append(
                GUIStepExample(
                    example_id=str(step.get("papo_step_id") or f"{episode_id}__{index:04d}"),
                    user_id=str(step.get("user_id") or task.get("input", {}).get("user_id", "")),
                    task_id=task_id,
                    instruction=instruction,
                    state_id=str(step.get("papo_step_id") or step.get("state_key") or f"{episode_id}__{index:04d}"),
                    source_trajectory_id=trajectory_id,
                    screenshot_path=str(step.get("screenshot_path") or task.get("input", {}).get("initial_screenshot") or "") or None,
                    ui_tree=_step_ui_tree(step),
                    action_history=action_history,
                    current_action=current_action,
                    future_trajectory=future_trajectory,
                    task_success=episode_success,
                    progress=(index + 1) / max(num_steps, 1),
                    goal_state=goal_state,
                    invalid_count=0 if bool(step.get("valid_observation")) else 1,
                    risk_score=max(0.0, 1.0 - float(step.get("action_confidence", 0.0) or 0.0)),
                    split=split,
                )
            )
    return examples


def _build_train_trajectories(tasks: list[dict[str, Any]], steps_by_episode: dict[str, list[dict[str, Any]]]) -> list[TrajectoryRecord]:
    trajectories: list[TrajectoryRecord] = []
    seen: set[str] = set()
    for task in tasks:
        episode_id = _episode_id_from_task(task)
        if episode_id in seen:
            continue
        seen.add(episode_id)
        instruction = _task_instruction(task)
        episode_steps = steps_by_episode.get(episode_id, [])
        if not episode_steps:
            raise ValueError(f"Missing PAPO steps for target episode {episode_id}; refusing to fall back to evaluation-only target.actions.")
        trajectories.append(_trajectory_from_episode(episode_id, episode_steps, instruction=instruction, split="train"))
    return trajectories


def _build_history_trajectories(
    tasks: list[dict[str, Any]],
    steps_by_episode: dict[str, list[dict[str, Any]]],
    existing_ids: set[str],
) -> list[TrajectoryRecord]:
    train_target_ids = {_episode_id_from_task(task) for task in tasks if str(task.get("metadata", {}).get("partition", "")).lower() != "eval"}
    eval_target_ids = {_episode_id_from_task(task) for task in tasks if str(task.get("metadata", {}).get("partition", "")).lower() == "eval"}
    referenced_payloads = _reference_payloads(tasks)
    trajectories: list[TrajectoryRecord] = []

    for episode_id, payload in sorted(referenced_payloads.items()):
        if episode_id in train_target_ids or episode_id in eval_target_ids:
            continue
        trajectory_id = f"episode::{episode_id}"
        if trajectory_id in existing_ids:
            continue
        episode_steps = steps_by_episode.get(episode_id, [])
        if episode_steps:
            trajectories.append(
                _trajectory_from_episode(
                    episode_id,
                    episode_steps,
                    instruction=str(payload.get("intent") or episode_steps[0].get("intent") or ""),
                    split="history",
                )
            )
            existing_ids.add(trajectory_id)
            continue

        actions = [parse_action_label(action) for action in payload.get("actions", [])]
        if not actions:
            continue
        success = 0.9 if str(payload.get("_source_kind", "")).startswith("same_user") else 0.6
        risk = 0.05 if str(payload.get("_source_kind", "")).startswith("same_user") else 0.15
        trajectories.append(
            TrajectoryRecord(
                trajectory_id=trajectory_id,
                user_id=str(payload.get("user_id", "")),
                task_id=stable_task_id(str(payload.get("intent", ""))),
                instruction=str(payload.get("intent", "")),
                actions=actions,
                task_success=success,
                progress=1.0,
                goal_state=str(payload.get("scenario") or payload.get("intent") or ""),
                invalid_count=0,
                risk_score=risk,
                split="history",
            )
        )
        existing_ids.add(trajectory_id)
    return trajectories


def _trajectory_from_episode(episode_id: str, episode_steps: list[dict[str, Any]], instruction: str, split: str) -> TrajectoryRecord:
    ordered = sorted(episode_steps, key=lambda row: int(row.get("step_index", 0) or 0))
    actions = [_parse_step_action(step) for step in ordered]
    risk_values = [max(0.0, 1.0 - float(step.get("action_confidence", 0.0) or 0.0)) for step in ordered]
    return TrajectoryRecord(
        trajectory_id=f"episode::{episode_id}",
        user_id=str(ordered[0].get("user_id", "")),
        task_id=stable_task_id(instruction or str(ordered[0].get("intent", ""))),
        instruction=instruction or str(ordered[0].get("intent", "")),
        actions=actions,
        states=[str(step.get("state_key") or "") for step in ordered],
        task_success=1.0 if bool(ordered[-1].get("is_terminal")) else 0.75,
        progress=1.0 if ordered else 0.0,
        goal_state=str(ordered[-1].get("next_state_key") or ordered[-1].get("state_key") or instruction),
        invalid_count=sum(0 if bool(step.get("valid_observation")) else 1 for step in ordered),
        risk_score=sum(risk_values) / max(len(risk_values), 1),
        split=split,
    )


def _parse_step_action(step: dict[str, Any]) -> Action:
    return parse_action_label(
        str(step.get("action") or step.get("raw_action") or ""),
        raw_action=str(step.get("raw_action") or ""),
        object_text=str(step.get("object_text") or ""),
        object_role=str(step.get("object_role") or ""),
    )


def _extract_input_text(raw_action: str | None) -> str:
    text = str(raw_action or "")
    if not text:
        return ""
    for field in ("content", "text"):
        single = re.search(field + r"\s*=\s*'([^']*)'", text)
        if single:
            return single.group(1).strip()
        double = re.search(field + r'\s*=\s*"([^"]*)"', text)
        if double:
            return double.group(1).strip()
    return ""


def _validate_target_episode_coverage(
    train_tasks: list[dict[str, Any]],
    eval_tasks: list[dict[str, Any]],
    steps_by_episode: dict[str, list[dict[str, Any]]],
) -> None:
    missing_train = sorted({episode_id for episode_id in (_episode_id_from_task(task) for task in train_tasks) if not steps_by_episode.get(episode_id)})
    missing_eval = sorted({episode_id for episode_id in (_episode_id_from_task(task) for task in eval_tasks) if not steps_by_episode.get(episode_id)})
    if not missing_train and not missing_eval:
        return

    details: list[str] = []
    if missing_train:
        details.append("train missing PAPO steps: " + ", ".join(missing_train[:10]))
    if missing_eval:
        details.append("eval missing PAPO steps: " + ", ".join(missing_eval[:10]))
    raise ValueError(
        "Main-project TN-DPO adapter requires PAPO step coverage for every target episode and will not backfill from evaluation-only target.actions.\n"
        + "\n".join(details)
    )


def _step_ui_tree(step: dict[str, Any]) -> str | None:
    tokens = [str(token) for token in step.get("object_tokens", []) if str(token).strip()]
    if tokens:
        return "\n".join(tokens[:96])
    return _safe_xml_text(step.get("xml_path"))


def _safe_xml_text(path_like: Any, max_chars: int = 4000) -> str | None:
    if not path_like:
        return None
    path = Path(str(path_like))
    if not path.is_file():
        return str(path_like)
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except OSError:
        return str(path_like)


def _goal_state(task: dict[str, Any], episode_steps: list[dict[str, Any]]) -> str:
    target = task.get("target") if isinstance(task.get("target"), dict) else {}
    if target.get("intent_class"):
        return str(target["intent_class"])
    if episode_steps:
        return str(episode_steps[-1].get("next_state_key") or episode_steps[-1].get("state_key") or "")
    return _task_instruction(task)


def _task_instruction(task: dict[str, Any]) -> str:
    return str(task.get("input", {}).get("instruction") or task.get("metadata", {}).get("intent") or "")


def _episode_id_from_task(task: dict[str, Any]) -> str:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    if metadata.get("papo_episode_id"):
        return str(metadata["papo_episode_id"])
    inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
    return f"{inputs.get('user_id', '')}__{inputs.get('time', '')}"


def _group_steps_by_episode(steps: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        grouped[str(step.get("episode_id") or "")].append(step)
    for episode_steps in grouped.values():
        episode_steps.sort(key=lambda row: int(row.get("step_index", 0) or 0))
    return dict(grouped)


def _reference_payloads(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for task in tasks:
        inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
        for key in ["same_user_action_references", "cross_user_action_references"]:
            values = inputs.get(key)
            if not isinstance(values, list):
                continue
            for payload in values:
                if not isinstance(payload, dict):
                    continue
                episode_id = str(payload.get("episode_id") or f"{payload.get('user_id', '')}__{payload.get('time', '')}")
                if episode_id not in payloads:
                    payloads[episode_id] = {**payload, "_source_kind": key.replace("_action_references", "")}
        for key in ["same_user_action_reference", "cross_user_action_reference"]:
            payload = inputs.get(key)
            if not isinstance(payload, dict):
                continue
            episode_id = str(payload.get("episode_id") or f"{payload.get('user_id', '')}__{payload.get('time', '')}")
            payloads.setdefault(episode_id, {**payload, "_source_kind": key.replace("_action_reference", "")})
    return payloads
