from __future__ import annotations

import copy
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .proactive_quality_gate import (
    ProactiveQualityGate,
    assistant_text,
    is_oracle_row,
    normalize_text,
    prompt_text,
    row_group_id,
    row_metadata,
    row_source,
    row_target,
    row_weight,
    summarize_numbers,
)


ORACLE_SOURCES = {"oracle", "oracle_target", "target", "gold", "ground_truth"}
HISTORY_SOURCES = {"same_user_history", "history", "user_history"}
SFT_SOURCES = {"sft_sample", "model_sample", "qwen_sft", "ui_tars_sft"}
CROSS_USER_SOURCES = {"cross_user_hard", "cross_user", "other_user"}


@dataclass
class CleanConfig:
    oracle_weight: float = 0.80
    min_oracle_margin: float = 0.10
    max_negatives_per_group: int = 3
    max_same_answer_frequency: int = 100
    allow_history_negatives: bool = False
    min_answer_chars: int = 2


@dataclass
class CleanArtifacts:
    listwise_rows: list[dict[str, Any]]
    dpo_rows: list[dict[str, Any]]
    rejected_rows: list[dict[str, Any]]
    report: dict[str, Any]


def clean_preference_split(
    rows: list[dict[str, Any]],
    *,
    split: str,
    config: CleanConfig,
) -> CleanArtifacts:
    groups = _group_listwise_rows(rows)
    answer_frequency = _global_non_oracle_answer_frequency(groups)
    rejected: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    dpo_rows: list[dict[str, Any]] = []
    before_source_counts: Counter[str] = Counter()
    after_source_counts: Counter[str] = Counter()
    before_answer_counts: Counter[str] = Counter()
    after_answer_counts: Counter[str] = Counter()
    kept_group_count = 0
    dropped_group_count = 0
    oracle_weights: list[float] = []
    margins: list[float] = []
    non_oracle_masses: list[float] = []
    reject_reasons: Counter[str] = Counter()

    for group_id, group_items in groups.items():
        group_rows = [row for _, row in group_items]
        for row in group_rows:
            before_source_counts[row_source(row)] += 1
            answer = normalize_text(assistant_text(row))
            if answer:
                before_answer_counts[answer] += 1

        oracle = _select_oracle(group_rows)
        if oracle is None:
            dropped_group_count += 1
            _reject_group(rejected, split, group_id, group_rows, "missing_oracle")
            reject_reasons["missing_oracle"] += len(group_rows)
            continue

        negatives: list[dict[str, Any]] = []
        prompt_norm = normalize_text(prompt_text(group_rows[0]))
        for row in group_rows:
            if row is oracle:
                continue
            reason = _negative_reject_reason(row, prompt_norm, answer_frequency, config)
            if reason:
                rejected.append(_rejected_row(split, group_id, row, reason))
                reject_reasons[reason] += 1
                continue
            negatives.append(row)

        negatives = _select_negatives(negatives, config.max_negatives_per_group)
        if not negatives:
            dropped_group_count += 1
            _reject_group(rejected, split, group_id, group_rows, "no_valid_negatives")
            reject_reasons["no_valid_negatives"] += len(group_rows)
            continue

        weighted = _assign_v3_weights(oracle, negatives, config.oracle_weight)
        oracle_v3_weight = weighted[0][1]
        max_negative_weight = max(weight for _, weight in weighted[1:])
        margin = oracle_v3_weight - max_negative_weight
        if margin < config.min_oracle_margin:
            dropped_group_count += 1
            _reject_group(
                rejected,
                split,
                group_id,
                group_rows,
                f"weak_v3_margin:{margin:.6f}",
            )
            reject_reasons["weak_v3_margin"] += len(group_rows)
            continue

        kept_group_count += 1
        oracle_weights.append(oracle_v3_weight)
        margins.append(margin)
        non_oracle_masses.append(1.0 - oracle_v3_weight)

        v3_rows: list[dict[str, Any]] = []
        for rank, (row, weight) in enumerate(weighted):
            clean_row = _clone_with_v3_metadata(row, split, group_id, weight, rank)
            v3_rows.append(clean_row)
            clean_rows.append(clean_row)
            after_source_counts[row_source(clean_row)] += 1
            answer = normalize_text(assistant_text(clean_row))
            if answer:
                after_answer_counts[answer] += 1

        dpo_rows.extend(_build_dpo_rows_from_group(v3_rows, split, group_id))

    report = {
        "split": split,
        "config": asdict(config),
        "before": {
            "rows": len(rows),
            "groups": len(groups),
            "source_counts": dict(before_source_counts),
            "unique_answers": len(before_answer_counts),
            "top_answers": _top_answers(before_answer_counts),
        },
        "after": {
            "listwise_rows": len(clean_rows),
            "dpo_rows": len(dpo_rows),
            "kept_groups": kept_group_count,
            "dropped_groups": dropped_group_count,
            "source_counts": dict(after_source_counts),
            "unique_answers": len(after_answer_counts),
            "top_answers": _top_answers(after_answer_counts),
            "oracle_weight": summarize_numbers(oracle_weights),
            "oracle_margin": summarize_numbers(margins),
            "non_oracle_mass": summarize_numbers(non_oracle_masses),
        },
        "rejected": {
            "rows": len(rejected),
            "reason_counts": dict(reject_reasons),
        },
    }
    return CleanArtifacts(clean_rows, dpo_rows, rejected, report)


def run_quality_gate_on_clean_artifacts(
    artifacts: CleanArtifacts,
    *,
    split: str,
    config: CleanConfig,
    image_roots: list[Path] | None = None,
) -> dict[str, Any]:
    gate = ProactiveQualityGate(
        image_roots=image_roots or [],
        min_oracle_margin=config.min_oracle_margin,
        max_answer_frequency=config.max_same_answer_frequency,
        max_non_oracle_mass=max(0.0, 1.0 - config.oracle_weight + 1e-9),
        leak_weight_threshold=0.05,
        progress_every=0,
    )
    summaries = [
        gate.audit_listwise(artifacts.listwise_rows, name=f"{split}_listwise_v3"),
        gate.audit_dpo(artifacts.dpo_rows, name=f"{split}_dpo_v3"),
    ]
    decision = gate.decide(summaries)
    return {
        "status": decision.status,
        "blocking_reasons": decision.blocking_reasons,
        "warning_reasons": decision.warning_reasons,
        "summaries": summaries,
        "issue_counts": {
            "by_severity": dict(Counter(issue.severity for issue in gate.issues)),
            "by_category": dict(Counter(issue.category for issue in gate.issues)),
        },
        "issues": [asdict(issue) for issue in gate.issues],
    }


def update_dataset_info_with_v3(dataset_info_path: Path) -> None:
    if dataset_info_path.exists():
        data = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    else:
        data = {}

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
    dpo = {
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
    data.update(
        {
            "papo_proactive_train_listwise_v3": {
                "file_name": "papo_proactive_train_listwise_v3.json",
                **mllm,
                "columns": {**mllm["columns"], "listwise_weight": "papo_listwise_weight"},
            },
            "papo_proactive_eval_listwise_v3": {
                "file_name": "papo_proactive_eval_listwise_v3.json",
                **mllm,
                "columns": {**mllm["columns"], "listwise_weight": "papo_listwise_weight"},
            },
            "papo_proactive_train_dpo_v3": {
                "file_name": "papo_proactive_train_dpo_v3.json",
                **dpo,
            },
            "papo_proactive_eval_dpo_v3": {
                "file_name": "papo_proactive_eval_dpo_v3.json",
                **dpo,
            },
        }
    )
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_info_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _group_listwise_rows(rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row_group_id(row, index)].append((index, row))
    return groups


def _global_non_oracle_answer_frequency(
    groups: dict[str, list[tuple[int, dict[str, Any]]]]
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for group_items in groups.values():
        rows = [row for _, row in group_items]
        oracle = _select_oracle(rows)
        for row in rows:
            if row is oracle:
                continue
            answer = normalize_text(assistant_text(row))
            if answer:
                counts[answer] += 1
    return counts


def _select_oracle(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    target = row_target(rows[0]) if rows else ""
    oracle_rows = [row for row in rows if is_oracle_row(row, assistant_text(row), target)]
    if not oracle_rows:
        return None
    return max(oracle_rows, key=row_weight)


def _negative_reject_reason(
    row: dict[str, Any],
    prompt_norm: str,
    answer_frequency: Counter[str],
    config: CleanConfig,
) -> str:
    answer = assistant_text(row)
    answer_norm = normalize_text(answer)
    source = row_source(row).lower()
    if not answer_norm or len(answer_norm) < config.min_answer_chars:
        return "invalid_or_too_short"
    if "\ufffd" in answer:
        return "unicode_replacement_char"
    if answer_norm in prompt_norm:
        return "prompt_history_copy"
    if source in HISTORY_SOURCES and not config.allow_history_negatives:
        return "history_negative_disabled"
    if answer_frequency[answer_norm] > config.max_same_answer_frequency:
        return "popular_answer_over_cap"
    return ""


def _select_negatives(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    def priority(row: dict[str, Any]) -> tuple[int, float, str]:
        source = row_source(row).lower()
        if source in SFT_SOURCES:
            source_rank = 0
        elif source in CROSS_USER_SOURCES:
            source_rank = 1
        elif source in HISTORY_SOURCES:
            source_rank = 2
        else:
            source_rank = 3
        return (source_rank, -row_weight(row), assistant_text(row))

    unique: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=priority):
        key = normalize_text(assistant_text(row))
        if key and key not in unique:
            unique[key] = row
    return list(unique.values())[:limit]


def _assign_v3_weights(
    oracle: dict[str, Any],
    negatives: list[dict[str, Any]],
    oracle_weight: float,
) -> list[tuple[dict[str, Any], float]]:
    oracle_weight = min(max(oracle_weight, 0.5), 0.99)
    negative_mass = 1.0 - oracle_weight
    if not negatives:
        return [(oracle, 1.0)]
    raw = [max(row_weight(row), 0.0) for row in negatives]
    if sum(raw) <= 0:
        weights = [negative_mass / len(negatives)] * len(negatives)
    else:
        total = sum(raw)
        weights = [negative_mass * value / total for value in raw]
    return [(oracle, oracle_weight), *list(zip(negatives, weights))]


def _clone_with_v3_metadata(
    row: dict[str, Any],
    split: str,
    group_id: str,
    weight: float,
    rank: int,
) -> dict[str, Any]:
    cloned = copy.deepcopy(row)
    metadata = row_metadata(cloned)
    cloned["metadata"] = metadata
    metadata.update(
        {
            "preference_version": "clean_v3",
            "preference_split": split,
            "preference_group_id": group_id,
            "v3_rank": rank,
            "v3_original_weight": row_weight(row),
            "v3_cleaning_policy": "oracle_anchor_prompt_leak_filter_frequency_cap_topk",
        }
    )
    cloned["papo_listwise_weight"] = float(weight)
    return cloned


def _build_dpo_rows_from_group(rows: list[dict[str, Any]], split: str, group_id: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    target = row_target(rows[0])
    oracle_rows = [row for row in rows if is_oracle_row(row, assistant_text(row), target)]
    if not oracle_rows:
        return []
    chosen = max(oracle_rows, key=row_weight)
    negatives = [row for row in rows if row is not chosen]
    negatives.sort(key=row_weight, reverse=True)
    out: list[dict[str, Any]] = []
    for rejected in negatives[:2]:
        margin = row_weight(chosen) - row_weight(rejected)
        out.append(
            {
                "conversations": _prompt_messages_for_dpo(chosen),
                "chosen": {"from": "gpt", "value": assistant_text(chosen)},
                "rejected": {"from": "gpt", "value": assistant_text(rejected)},
                "images": list(chosen.get("images") or []),
                "papo_weight": float(max(margin, 0.0)),
                "papo_target_probability": float(row_weight(chosen)),
                "metadata": {
                    "preference_version": "clean_v3",
                    "preference_split": split,
                    "preference_group_id": group_id,
                    "chosen_source": row_source(chosen),
                    "rejected_source": row_source(rejected),
                    "chosen_weight": row_weight(chosen),
                    "rejected_weight": row_weight(rejected),
                    "preference_margin": margin,
                    "target": target,
                },
            }
        )
    return out


def _prompt_messages_for_dpo(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages") or row.get("conversations") or []
    out: list[dict[str, str]] = []
    if not isinstance(messages, list):
        return out
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or message.get("from") or "").lower()
        content = str(message.get("content") or message.get("value") or "")
        if role in {"assistant", "gpt", "model"}:
            continue
        if role == "system":
            out.append({"from": "system", "value": content})
        else:
            out.append({"from": "human", "value": content})
    return out


def _reject_group(
    rejected: list[dict[str, Any]],
    split: str,
    group_id: str,
    rows: list[dict[str, Any]],
    reason: str,
) -> None:
    for row in rows:
        rejected.append(_rejected_row(split, group_id, row, reason))


def _rejected_row(split: str, group_id: str, row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "split": split,
        "group_id": group_id,
        "reason": reason,
        "source": row_source(row),
        "weight": row_weight(row),
        "answer": assistant_text(row),
        "target": row_target(row),
    }


def _top_answers(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
    return [{"answer": answer, "count": count} for answer, count in counter.most_common(limit)]
