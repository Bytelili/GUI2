from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a personalized Android GUI agent. Follow the requested output format exactly. "
    "Use user history only when it is relevant and never reveal hidden target fields."
)


def export_preference_datasets(
    scored_sets: list[dict[str, Any]],
    *,
    raw_root: str | Path,
    asset_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    listwise: list[dict[str, Any]] = []
    dpo: list[dict[str, Any]] = []
    for row in scored_sets:
        prompt = proactive_prompt(row["input"])
        images = [
            _relative_asset(str(path), raw_root, asset_prefix)
            for path in row["input"].get("initial_screenshots", [])
            if str(path)
        ]
        metadata = _base_metadata(row)
        for candidate in row["candidates"]:
            probability = float(candidate.get("target_policy_probability", 0.0) or 0.0)
            if probability <= 0.0:
                continue
            listwise.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _image_prefix(len(images)) + prompt},
                        {"role": "assistant", "content": str(candidate.get("text") or "")},
                    ],
                    "images": images,
                    "papo_listwise_weight": probability,
                    "metadata": {
                        **metadata,
                        "candidate_id": candidate.get("candidate_id", ""),
                        "candidate_source": candidate.get("source", ""),
                        "candidate_source_episode_id": candidate.get("source_episode_id", ""),
                        "reward": candidate.get("reward", {}),
                        "target_policy_probability": probability,
                    },
                }
            )
        for pair in row["pairs"]:
            dpo.append(
                {
                    "conversations": [
                        {"from": "system", "value": SYSTEM_PROMPT},
                        {"from": "human", "value": _image_prefix(len(images)) + prompt},
                    ],
                    "chosen": {"from": "gpt", "value": pair["chosen"]},
                    "rejected": {"from": "gpt", "value": pair["rejected"]},
                    "images": images,
                    "papo_weight": float(pair["weight"]),
                    "papo_target_probability": float(pair["target_preference_probability"]),
                    "metadata": {
                        **metadata,
                        "chosen_candidate_id": pair["chosen_candidate_id"],
                        "rejected_candidate_id": pair["rejected_candidate_id"],
                        "chosen_source": pair["chosen_source"],
                        "rejected_source": pair["rejected_source"],
                        "reward_gap": pair["reward_gap"],
                    },
                }
            )
    return listwise, dpo


def preference_dataset_info() -> dict[str, Any]:
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
        result[f"papo_proactive_{partition}_listwise"] = {
            "file_name": f"papo_proactive_{partition}_listwise.json",
            **mllm,
            "columns": {
                **mllm["columns"],
                "listwise_weight": "papo_listwise_weight",
            },
        }
        result[f"papo_proactive_{partition}_dpo"] = {
            "file_name": f"papo_proactive_{partition}_dpo.json",
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


def _base_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "papo_episode_id": metadata.get("papo_episode_id", ""),
        "partition": row.get("partition", ""),
        "protocol_id": metadata.get("protocol_id", ""),
        "target_split": metadata.get("target_split", ""),
        "history_split": metadata.get("history_split", ""),
        "history_policy": metadata.get("history_policy", ""),
        "history_episode_ids": list(metadata.get("history_episode_ids") or []),
        "candidate_reference_partition": row.get("candidate_reference_partition", ""),
    }


def _relative_asset(path: str, raw_root: str | Path, asset_prefix: str) -> str:
    source = Path(path)
    try:
        relative = source.resolve().relative_to(Path(raw_root).resolve()).as_posix()
        return f"{asset_prefix.strip('/')}/{relative}"
    except ValueError:
        return source.as_posix()


def _image_prefix(count: int) -> str:
    return "<image>" * count
