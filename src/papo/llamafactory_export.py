from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl, step_id


SYSTEM_PROMPT = (
    "You are a personalized Android GUI agent. Follow the requested output format exactly. "
    "Use user history only when it is relevant and never reveal hidden target fields."
)


def export_proactive_sft(
    tasks: list[dict[str, Any]], raw_root: str | Path, asset_prefix: str = "RawDataset"
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        inputs = task.get("input", {})
        target = task.get("target", {})
        images = [_relative_asset(path, raw_root, asset_prefix) for path in inputs.get("initial_screenshots", []) if path]
        prompt = _image_prefix(len(images)) + proactive_prompt(inputs)
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": str(target.get("intent") or "")},
                ],
                "images": images,
                "metadata": _export_metadata(task),
            }
        )
    return rows


def export_execution_sft(
    tasks: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    raw_root: str | Path,
    asset_prefix: str = "RawDataset",
) -> list[dict[str, Any]]:
    task_by_episode = {_task_episode_id(task): task for task in tasks}
    rows: list[dict[str, Any]] = []
    for step in steps:
        task = task_by_episode.get(str(step.get("episode_id") or ""))
        if task is None or not step.get("valid_observation"):
            continue
        screenshot = str(step.get("screenshot_path") or "")
        images = [_relative_asset(screenshot, raw_root, asset_prefix)] if screenshot else []
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _image_prefix(len(images)) + _execution_prompt(task, step)},
                    {"role": "assistant", "content": str(step.get("raw_action") or step.get("action") or "")},
                ],
                "images": images,
                "metadata": _export_metadata(task, step),
            }
        )
    return rows


def export_execution_dpo(
    tasks: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    raw_root: str | Path,
    asset_prefix: str = "RawDataset",
) -> list[dict[str, Any]]:
    task_by_episode = {_task_episode_id(task): task for task in tasks}
    step_by_id = {step_id(step): step for step in steps}
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        step = step_by_id.get(str(pair.get("step_id") or ""))
        task = task_by_episode.get(str(pair.get("episode_id") or ""))
        if step is None or task is None or not step.get("valid_observation"):
            continue
        screenshot = str(step.get("screenshot_path") or "")
        images = [_relative_asset(screenshot, raw_root, asset_prefix)] if screenshot else []
        rows.append(
            {
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {
                        "from": "human",
                        "value": _image_prefix(len(images))
                        + _execution_prompt(task, step, list(pair.get("prefix_actions") or [])),
                    },
                ],
                "chosen": {"from": "gpt", "value": str(pair.get("positive_action") or "")},
                "rejected": {"from": "gpt", "value": str(pair.get("negative_action") or "")},
                "images": images,
                "papo_weight": float(pair.get("weight", 1.0) or 1.0),
                "papo_target_probability": float(pair.get("target_preference_probability", 1.0) or 1.0),
                "metadata": {
                    **_export_metadata(task, step),
                    "advantage_gap": pair.get("advantage_gap", 0.0),
                    "weight": pair.get("weight", 1.0),
                    "target_preference_probability": pair.get("target_preference_probability", 1.0),
                    "positive_source": pair.get("positive_source", ""),
                    "negative_source": pair.get("negative_source", ""),
                    "state_key": pair.get("state_key", ""),
                },
            }
        )
    return rows


def export_execution_listwise(
    tasks: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    listwise_targets: list[dict[str, Any]],
    raw_root: str | Path,
    asset_prefix: str = "RawDataset",
) -> list[dict[str, Any]]:
    task_by_episode = {_task_episode_id(task): task for task in tasks}
    step_by_id = {step_id(step): step for step in steps}
    rows: list[dict[str, Any]] = []
    for state in listwise_targets:
        step = step_by_id.get(str(state.get("step_id") or ""))
        task = task_by_episode.get(str(state.get("episode_id") or ""))
        if step is None or task is None or not step.get("valid_observation"):
            continue
        screenshot = str(step.get("screenshot_path") or "")
        images = [_relative_asset(screenshot, raw_root, asset_prefix)] if screenshot else []
        for candidate in state.get("candidates", []):
            probability = float(candidate.get("target_policy_probability", 0.0) or 0.0)
            if probability <= 0.0:
                continue
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _image_prefix(len(images))
                            + _execution_prompt(task, step, list(state.get("prefix_actions") or [])),
                        },
                        {"role": "assistant", "content": str(candidate.get("action") or "")},
                    ],
                    "images": images,
                    "papo_listwise_weight": probability,
                    "metadata": {
                        **_export_metadata(task, step),
                        "tree_id": state.get("tree_id", ""),
                        "node_id": state.get("node_id", ""),
                        "base_policy_probability": candidate.get("base_policy_probability", 0.0),
                        "target_policy_probability": probability,
                        "a_delta": candidate.get("a_delta", 0.0),
                    },
                }
            )
    return rows


def write_json(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")


def dataset_info() -> dict[str, Any]:
    mllm = {
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    result: dict[str, Any] = {}
    for partition in ["train", "eval"]:
        result[f"papo_proactive_{partition}_sft"] = {
            "file_name": f"papo_proactive_{partition}_sft.json",
            **mllm,
        }
        result[f"papo_execution_{partition}_sft"] = {
            "file_name": f"papo_execution_{partition}_sft.json",
            **mllm,
        }
        result[f"papo_execution_{partition}_listwise"] = {
            "file_name": f"papo_execution_{partition}_listwise.json",
            **mllm,
            "columns": {
                **mllm["columns"],
                "listwise_weight": "papo_listwise_weight",
            },
        }
        result[f"papo_execution_{partition}_dpo"] = {
            "file_name": f"papo_execution_{partition}_dpo.json",
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
                "images": "images",
                "preference_weight": "papo_weight",
                "preference_target": "papo_target_probability",
            },
        }
    return result


def proactive_prompt(inputs: dict[str, Any]) -> str:
    history = [
        f"- {item.get('time', '')} | {item.get('scenario', '')} | {item.get('intent', '')}"
        for item in inputs.get("previous_intents", [])
    ]
    return "\n".join(
        [
            "Infer the user's current intent. Output exactly one Chinese sentence.",
            f"Time: {inputs.get('time', '')}",
            f"Scenario: {inputs.get('scenario', '')}",
            f"User profile: {json.dumps(inputs.get('user_profile', {}), ensure_ascii=False)}",
            "Previous intents:",
            *(history or ["- none"]),
        ]
    )


def _execution_prompt(
    task: dict[str, Any], step: dict[str, Any], prior_actions_override: list[str] | None = None
) -> str:
    inputs = task.get("input", {})
    same_refs = inputs.get("same_user_action_references") or [inputs.get("same_user_action_reference") or {}]
    reference_actions = [ref.get("actions", []) for ref in same_refs if isinstance(ref, dict)]
    prior_actions = (
        prior_actions_override
        if prior_actions_override is not None
        else task.get("_prior_actions", {}).get(str(step.get("step_index") or 0), [])
    )
    return "\n".join(
        [
            "Predict exactly one next Android action and output no explanation.",
            f"Instruction: {inputs.get('instruction', '')}",
            f"Scenario: {inputs.get('scenario', '')}",
            f"User profile: {json.dumps(inputs.get('user_profile', {}), ensure_ascii=False)}",
            f"Relevant same-user reference actions: {json.dumps(reference_actions, ensure_ascii=False)}",
            f"Previous actions: {json.dumps(prior_actions, ensure_ascii=False)}",
            f"Current UI elements: {json.dumps(step.get('object_tokens', []), ensure_ascii=False)}",
            "Allowed actions: click, long_click, type, scroll, press_back, press_home, press_recent, wait, finished.",
        ]
    )


def attach_prior_actions(tasks: list[dict[str, Any]], steps: list[dict[str, Any]]) -> None:
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for step in steps:
        by_episode.setdefault(str(step.get("episode_id") or ""), []).append(step)
    task_by_episode = {_task_episode_id(task): task for task in tasks}
    for episode_id, episode_steps in by_episode.items():
        task = task_by_episode.get(episode_id)
        if task is None:
            continue
        episode_steps.sort(key=lambda item: int(item.get("step_index", 0) or 0))
        task["_prior_actions"] = {}
        history: list[str] = []
        for step in episode_steps:
            task["_prior_actions"][str(step.get("step_index") or 0)] = list(history)
            history.append(str(step.get("raw_action") or step.get("action") or ""))


def _relative_asset(path: str, raw_root: str | Path, asset_prefix: str) -> str:
    source = Path(path)
    try:
        relative = source.resolve().relative_to(Path(raw_root).resolve()).as_posix()
        return f"{asset_prefix.strip('/')}/{relative}"
    except ValueError:
        return source.as_posix()


def _image_prefix(count: int) -> str:
    return "<image>" * count


def _task_episode_id(task: dict[str, Any]) -> str:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    direct = str(metadata.get("papo_episode_id") or "")
    if direct:
        return direct
    task_id = str(task.get("task_id") or "")
    for prefix in ["execution__", "suggestion__"]:
        if task_id.startswith(prefix):
            return task_id[len(prefix):]
    inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
    return f"{inputs.get('user_id', '')}__{inputs.get('time', '')}"


def _export_metadata(task: dict[str, Any], step: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    exported = {
        key: metadata.get(key)
        for key in [
            "papo_episode_id",
            "partition",
            "protocol_id",
            "target_split",
            "history_split",
            "reference_split",
            "history_policy",
            "reference_policy",
            "history_episode_ids",
            "same_user_reference_episode_ids",
            "cross_user_reference_episode_ids",
        ]
        if key in metadata
    }
    if step is not None:
        exported["papo_step_id"] = step_id(step)
    return exported


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)
