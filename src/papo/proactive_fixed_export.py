from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from .io import write_json, write_jsonl
from .official_data import read_csv_rows
from .proactive_quality_gate import normalize_text


SFT_SYSTEM_PROMPT = (
    "You are a personalized Android GUI agent. Infer the user's current intent. "
    "Output exactly one Chinese sentence."
)

RERANK_SYSTEM_PROMPT = (
    "You are a personalized Android GUI intent reranker. Select the candidate that best matches "
    "the current screen, user context, and history. Output only the candidate letter."
)

SAME_USER_HARD_NEGATIVE = "same_user_hard_negative"
CONTEXT_HARD_NEGATIVE = "context_hard_negative"
SLOT_MISMATCH_SAME_USER = "slot_mismatch_same_user"

ORACLE_SOURCE = "oracle"
SAME_USER_SOURCE = "same_user"
CONTEXT_SOURCE = "context"


@dataclass(frozen=True)
class DPOExportConfig:
    max_pairs_per_row: int = 2
    min_reward_gap: float = 0.05
    min_char_similarity: float = 0.20
    max_char_similarity: float = 0.98
    same_user_min_similarity: float = 0.45
    same_user_max_similarity: float = 0.95
    context_min_similarity: float = 0.20
    context_max_similarity: float = 0.85
    same_user_min_semantic_similarity: float = 0.45
    context_min_semantic_similarity: float = 0.15


@dataclass(frozen=True)
class RerankExportConfig:
    min_candidates: int = 2
    shuffle_candidates: bool = True
    seed: int = 42


@dataclass(frozen=True)
class WeightedListwiseExportConfig:
    temperature: float = 0.15
    min_context_prob: float = 0.02
    min_oracle_prob: float = 0.65
    max_oracle_prob: float = 0.95
    max_same_user_prob: float = 0.25
    max_context_prob: float = 0.15
    context_hardness_threshold: float = 0.20


def read_wide_csv(path: str | Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    return [_normalize_wide_row(row) for row in rows]


def audit_wide_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    warnings: list[str] = []
    split_counts = Counter(str(row.get("split") or "") for row in rows)
    intent_class_counts = Counter(str(row.get("intent_class") or "") for row in rows)
    target_app_counts = Counter(str(row.get("target_app") or "") for row in rows)
    image_count_counts = Counter(int(row.get("image_count") or 0) for row in rows)
    history_count_counts = Counter(int(row.get("history_count") or 0) for row in rows)
    users = {str(row.get("user_id") or "") for row in rows if str(row.get("user_id") or "")}

    oracle_prob_values = [float(row.get("oracle_prob") or 0.0) for row in rows]
    same_user_prob_values = [float(row.get("same_user_prob") or 0.0) for row in rows]
    context_prob_values = [_float_or_nan(row.get("context_prob")) for row in rows]
    oracle_reward_values = [float(row.get("oracle_reward_total") or 0.0) for row in rows]
    same_user_reward_values = [float(row.get("same_user_reward_total") or 0.0) for row in rows]
    context_reward_values = [float(row.get("context_reward_total") or 0.0) for row in rows]
    oracle_margin_prob_values = [float(row.get("oracle_margin_prob") or 0.0) for row in rows]
    oracle_margin_reward_values = [float(row.get("oracle_margin_reward") or 0.0) for row in rows]
    same_user_semantic_values = [float(row.get("same_user_semantic_similarity") or 0.0) for row in rows]
    context_semantic_values = [float(row.get("context_semantic_similarity") or 0.0) for row in rows]
    dpo_rejected_count_values = [int(float(row.get("dpo_rejected_count") or 0.0)) for row in rows]

    same_user_source_match = _ratio(
        [
            row
            for row in rows
            if row.get("same_user_text") and row.get("same_user_source_app") and row.get("target_app")
        ],
        lambda row: str(row.get("same_user_source_app") or "") == str(row.get("target_app") or ""),
    )
    context_source_match = _ratio(
        [
            row
            for row in rows
            if row.get("context_text") and row.get("context_source_app") and row.get("target_app")
        ],
        lambda row: str(row.get("context_source_app") or "") == str(row.get("target_app") or ""),
    )

    same_user_eligibility = Counter(str(row.get("same_user_eligibility") or "") for row in rows)
    context_eligibility = Counter(str(row.get("context_eligibility") or "") for row in rows)
    release_eligibility = Counter(str(row.get("release_eligibility") or "") for row in rows)

    oracle_same_char_similarity_values: list[float] = []
    oracle_context_char_similarity_values: list[float] = []
    for row in rows:
        oracle_text = str(row.get("oracle_text") or "")
        same_user_text = str(row.get("same_user_text") or "")
        context_text = str(row.get("context_text") or "")
        if oracle_text and same_user_text:
            oracle_same_char_similarity_values.append(char_similarity(oracle_text, same_user_text))
        if oracle_text and context_text:
            oracle_context_char_similarity_values.append(char_similarity(oracle_text, context_text))

    if len({round(value, 12) for value in oracle_prob_values}) <= 1:
        warnings.append("oracle_prob has a single unique value")
    if len({round(value, 12) for value in same_user_prob_values}) <= 1:
        warnings.append("same_user_prob has a single unique value")
    if context_prob_values and all((not math.isfinite(value)) or abs(value) <= 1e-12 for value in context_prob_values):
        warnings.append("context_prob is all zero or NaN")
    if dpo_rejected_count_values and all(value == 0 for value in dpo_rejected_count_values):
        warnings.append("dpo_rejected_count is all zero")
    if split_counts and set(split_counts) == {"train"}:
        warnings.append("split is entirely train")
    if oracle_same_char_similarity_values and _percentile(oracle_same_char_similarity_values, 95) >= 0.95:
        warnings.append("same_user_text is often too similar to oracle_text")
    if rows and all((not row.get("context_text")) or float(row.get("context_prob") or 0.0) <= 0.0 for row in rows):
        warnings.append("context candidates never receive training probability")

    return {
        "status": "passed" if not warnings else "warning",
        "row_count": len(rows),
        "user_count": len(users),
        "split_distribution": dict(split_counts),
        "intent_class_distribution": dict(intent_class_counts),
        "target_app_distribution_top50": dict(target_app_counts.most_common(50)),
        "image_count_distribution": {str(key): value for key, value in sorted(image_count_counts.items())},
        "history_count_distribution": {str(key): value for key, value in sorted(history_count_counts.items())},
        "oracle_prob_distribution": summarize_numbers(oracle_prob_values),
        "same_user_prob_distribution": summarize_numbers(same_user_prob_values),
        "context_prob_distribution": summarize_numbers([value for value in context_prob_values if math.isfinite(value)]),
        "oracle_reward_total_distribution": summarize_numbers(oracle_reward_values),
        "same_user_reward_total_distribution": summarize_numbers(same_user_reward_values),
        "context_reward_total_distribution": summarize_numbers(context_reward_values),
        "oracle_margin_prob_distribution": summarize_numbers(oracle_margin_prob_values),
        "oracle_margin_reward_distribution": summarize_numbers(oracle_margin_reward_values),
        "same_user_semantic_similarity_distribution": summarize_numbers(same_user_semantic_values),
        "context_semantic_similarity_distribution": summarize_numbers(context_semantic_values),
        "same_user_source_app_equals_target_app_ratio": same_user_source_match,
        "context_source_app_equals_target_app_ratio": context_source_match,
        "same_user_eligibility_distribution": dict(same_user_eligibility),
        "context_eligibility_distribution": dict(context_eligibility),
        "release_eligibility_distribution": dict(release_eligibility),
        "dpo_rejected_count_distribution": dict(Counter(dpo_rejected_count_values)),
        "oracle_same_char_similarity_distribution": summarize_numbers(oracle_same_char_similarity_values),
        "oracle_context_char_similarity_distribution": summarize_numbers(oracle_context_char_similarity_values),
        "warnings": warnings,
    }


def split_rows_by_user_time(
    rows: list[dict[str, Any]],
    eval_ratio: float,
    split_by: str = "user_time",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if split_by != "user_time":
        raise ValueError(f"Unsupported split_by: {split_by}")
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError(f"eval_ratio must be in (0, 1), got {eval_ratio}")

    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_user[str(row.get("user_id") or "")].append(row)

    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    user_breakdown: dict[str, dict[str, int]] = {}
    for user_id, user_rows in by_user.items():
        ordered = sorted(
            user_rows,
            key=lambda item: (
                str(item.get("target_time") or ""),
                str(item.get("group_id") or item.get("task_id") or ""),
            ),
        )
        eval_count = 0
        if len(ordered) > 1:
            eval_count = max(1, int(math.ceil(len(ordered) * eval_ratio)))
            eval_count = min(eval_count, len(ordered) - 1)
        split_index = len(ordered) - eval_count
        train.extend(ordered[:split_index])
        eval_rows.extend(ordered[split_index:])
        user_breakdown[user_id] = {
            "total": len(ordered),
            "train": len(ordered[:split_index]),
            "eval": len(ordered[split_index:]),
        }

    train_group_ids = {str(row.get("group_id") or row.get("task_id") or "") for row in train}
    eval_group_ids = {str(row.get("group_id") or row.get("task_id") or "") for row in eval_rows}
    overlap = sorted(group_id for group_id in train_group_ids & eval_group_ids if group_id)
    if overlap:
        raise ValueError(f"group_id appears in both train and eval: {overlap[0]}")

    summary = {
        "split_by": split_by,
        "eval_ratio": eval_ratio,
        "train_rows": len(train),
        "eval_rows": len(eval_rows),
        "user_count": len(by_user),
        "user_breakdown_top20": dict(sorted(user_breakdown.items())[:20]),
        "group_overlap_count": len(overlap),
    }
    return train, eval_rows, summary


def export_oracle_sft_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result: list[dict[str, Any]] = []
    dropped = Counter()
    for row in rows:
        prompt = str(row.get("prompt_text") or "")
        oracle = str(row.get("oracle_text") or "")
        if not prompt:
            dropped["empty_prompt_text"] += 1
            continue
        if not oracle:
            dropped["empty_oracle_text"] += 1
            continue
        result.append(
            {
                "messages": [
                    {"role": "system", "content": SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": oracle},
                ],
                "images": list(row.get("image_paths", [])),
                "metadata": {
                    "task_id": row.get("task_id", ""),
                    "group_id": row.get("group_id", ""),
                    "user_id": row.get("user_id", ""),
                    "target_time": row.get("target_time", ""),
                    "intent_class": row.get("intent_class", ""),
                    "target_app": row.get("target_app", ""),
                    "source": "oracle_only_sft",
                },
            }
        )
    report = {
        "status": "passed",
        "rows": len(result),
        "dropped": dict(dropped),
        "user_count": len({row["metadata"]["user_id"] for row in result if row["metadata"]["user_id"]}),
        "intent_class_distribution": dict(Counter(row["metadata"]["intent_class"] for row in result)),
    }
    return result, report


def export_dpo_rows(
    rows: Iterable[dict[str, Any]],
    config: DPOExportConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result: list[dict[str, Any]] = []
    type_counts = Counter()
    reward_gaps: list[float] = []
    char_similarities: list[float] = []
    semantic_similarities: list[float] = []
    rejected = Counter()

    for row in rows:
        prompt = str(row.get("prompt_text") or "")
        oracle = str(row.get("oracle_text") or "")
        if not prompt or not oracle:
            rejected["missing_prompt_or_oracle"] += 1
            continue

        negatives = _candidate_dpo_negatives(row, config)
        if not negatives:
            rejected["no_valid_negative"] += 1
            continue

        for negative in negatives[: max(1, config.max_pairs_per_row)]:
            if normalize_text(oracle) == normalize_text(negative["text"]):
                rejected["chosen_equals_rejected"] += 1
                continue
            if len(normalize_text(negative["text"])) < 2:
                rejected["rejected_too_short"] += 1
                continue
            if negative["char_similarity"] < config.min_char_similarity:
                rejected["char_similarity_too_low"] += 1
                continue
            if negative["char_similarity"] > config.max_char_similarity:
                rejected["char_similarity_too_high"] += 1
                continue
            if negative["reward_gap"] < config.min_reward_gap:
                rejected["reward_gap_too_low"] += 1
                continue

            result.append(
                {
                    "conversations": [
                        {"role": "system", "content": SFT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "chosen": {"role": "assistant", "content": oracle},
                    "rejected": {"role": "assistant", "content": negative["text"]},
                    "images": list(row.get("image_paths", [])),
                    "papo_weight": negative["weight"],
                    "papo_target_probability": 1.0,
                    "metadata": {
                        "task_id": row.get("task_id", ""),
                        "group_id": row.get("group_id", ""),
                        "user_id": row.get("user_id", ""),
                        "target_time": row.get("target_time", ""),
                        "negative_type": negative["negative_type"],
                        "negative_source_time": negative["source_time"],
                        "negative_source_app": negative["source_app"],
                        "chosen_text": oracle,
                        "rejected_text": negative["text"],
                        "oracle_reward_total": float(row.get("oracle_reward_total") or 0.0),
                        "negative_reward_total": negative["negative_reward_total"],
                        "reward_gap": negative["reward_gap"],
                        "char_similarity": negative["char_similarity"],
                        "semantic_similarity": negative["semantic_similarity"],
                        "intent_class": row.get("intent_class", ""),
                        "target_app": row.get("target_app", ""),
                    },
                }
            )
            type_counts[negative["negative_type"]] += 1
            reward_gaps.append(negative["reward_gap"])
            char_similarities.append(negative["char_similarity"])
            semantic_similarities.append(negative["semantic_similarity"])

    report = {
        "status": "passed" if result else "warning",
        "rows": len(result),
        "negative_type_distribution": dict(type_counts),
        "reward_gap_distribution": summarize_numbers(reward_gaps),
        "char_similarity_distribution": summarize_numbers(char_similarities),
        "semantic_similarity_distribution": summarize_numbers(semantic_similarities),
        "average_char_similarity": _mean(char_similarities),
        "average_reward_gap": _mean(reward_gaps),
        "rejected": dict(rejected),
    }
    return result, report


def export_rerank_rows(
    rows: Iterable[dict[str, Any]],
    config: RerankExportConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result: list[dict[str, Any]] = []
    answer_counts = Counter()
    oracle_position_counts = Counter()
    candidate_count_counts = Counter()
    rejected = Counter()

    for row in rows:
        prompt = str(row.get("prompt_text") or "")
        oracle = str(row.get("oracle_text") or "")
        if not prompt or not oracle:
            rejected["missing_prompt_or_oracle"] += 1
            continue

        candidates = _build_rerank_candidates(row)
        if len(candidates) < config.min_candidates:
            rejected["too_few_candidates"] += 1
            continue

        ordered = list(candidates)
        if config.shuffle_candidates:
            random.Random(_stable_seed(str(row.get("task_id") or ""), config.seed)).shuffle(ordered)

        letters = _candidate_letters(len(ordered))
        correct_letter = ""
        candidate_lines: list[str] = []
        for index, candidate in enumerate(ordered):
            letter = letters[index]
            candidate_lines.append(f"{letter}. {candidate['text']}")
            if candidate["source"] == ORACLE_SOURCE:
                correct_letter = letter
                oracle_position_counts[letter] += 1
        if not correct_letter:
            rejected["missing_oracle_after_shuffle"] += 1
            continue

        answer_counts[correct_letter] += 1
        candidate_count_counts[len(ordered)] += 1

        result.append(
            {
                "messages": [
                    {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{prompt}\n\nCandidates:\n"
                            + "\n".join(candidate_lines)
                            + "\n\nAnswer with only "
                            + ", ".join(letters[: len(ordered) - 1])
                            + f", or {letters[len(ordered) - 1]}."
                        ),
                    },
                    {"role": "assistant", "content": correct_letter},
                ],
                "images": list(row.get("image_paths", [])),
                "metadata": {
                    "task_id": row.get("task_id", ""),
                    "group_id": row.get("group_id", ""),
                    "user_id": row.get("user_id", ""),
                    "correct_letter": correct_letter,
                    "candidate_order": [candidate["source"] for candidate in ordered],
                    "oracle_text": oracle,
                    "same_user_text": row.get("same_user_text", ""),
                    "context_text": row.get("context_text", ""),
                    "candidate_count": len(ordered),
                },
            }
        )

    letter_distribution = {key: value for key, value in sorted(answer_counts.items())}
    answer_share = {
        key: (value / len(result) if result else 0.0)
        for key, value in letter_distribution.items()
    }
    warnings = [
        f"answer letter {letter} exceeds 45%" for letter, share in answer_share.items() if share > 0.45
    ]
    report = {
        "status": "passed" if not warnings else "warning",
        "rows": len(result),
        "answer_letter_distribution": letter_distribution,
        "answer_letter_share": answer_share,
        "oracle_position_distribution": {key: value for key, value in sorted(oracle_position_counts.items())},
        "candidate_count_distribution": {str(key): value for key, value in sorted(candidate_count_counts.items())},
        "rejected": dict(rejected),
        "warnings": warnings,
    }
    return result, report


def export_weighted_listwise_rows(
    rows: Iterable[dict[str, Any]],
    config: WeightedListwiseExportConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result: list[dict[str, Any]] = []
    oracle_weights: list[float] = []
    same_user_weights: list[float] = []
    context_weights: list[float] = []
    rejected = Counter()
    fixed_weight_groups = 0
    total_groups = 0

    for row in rows:
        prompt = str(row.get("prompt_text") or "")
        oracle = str(row.get("oracle_text") or "")
        if not prompt or not oracle:
            rejected["missing_prompt_or_oracle"] += 1
            continue
        group_candidates = _build_weighted_candidates(row)
        if len(group_candidates) < 2:
            rejected["too_few_candidates"] += 1
            continue

        probs = _dynamic_candidate_probabilities(row, group_candidates, config)
        total_groups += 1
        if len({round(value, 6) for value in probs.values()}) == 1:
            fixed_weight_groups += 1

        for candidate in group_candidates:
            source = candidate["source"]
            probability = probs[source]
            if probability <= 0.0:
                continue
            if source == ORACLE_SOURCE:
                oracle_weights.append(probability)
            elif source == SAME_USER_SOURCE:
                same_user_weights.append(probability)
            elif source == CONTEXT_SOURCE:
                context_weights.append(probability)
            result.append(
                {
                    "messages": [
                        {"role": "system", "content": SFT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": candidate["text"]},
                    ],
                    "images": list(row.get("image_paths", [])),
                    "papo_listwise_weight": probability,
                    "metadata": {
                        "task_id": row.get("task_id", ""),
                        "group_id": row.get("group_id", ""),
                        "user_id": row.get("user_id", ""),
                        "target": oracle,
                        "candidate_source": source,
                        "candidate_reward": candidate["reward"],
                        "target_policy_probability": probability,
                    },
                }
            )

    report = {
        "status": "passed",
        "rows": len(result),
        "groups": total_groups,
        "fixed_weight_groups": fixed_weight_groups,
        "oracle_average_weight": _mean(oracle_weights),
        "same_user_average_weight": _mean(same_user_weights),
        "context_average_weight": _mean(context_weights),
        "context_non_zero_count": sum(1 for value in context_weights if value > 0.0),
        "rejected": dict(rejected),
    }
    return result, report


def validate_sft_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    leak_count = 0
    user_ids: Counter[str] = Counter()
    intent_classes: Counter[str] = Counter()
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            issues.append(f"sft[{index}] missing messages")
            continue
        user_prompt = str(messages[1].get("content") or "")
        assistant = str(messages[2].get("content") or "")
        if not assistant.strip():
            issues.append(f"sft[{index}] empty assistant content")
        if row.get("images") is None or not isinstance(row.get("images"), list):
            issues.append(f"sft[{index}] images is not a list")
        if normalize_text(assistant) and normalize_text(assistant) in normalize_text(user_prompt):
            leak_count += 1
        metadata = row.get("metadata", {})
        user_ids[str(metadata.get("user_id") or "")] += 1
        intent_classes[str(metadata.get("intent_class") or "")] += 1
    return {
        "rows": len(rows),
        "user_count": len([user for user in user_ids if user]),
        "intent_class_distribution": dict(intent_classes),
        "oracle_prompt_leak_count": leak_count,
        "issues": issues,
        "passed": not issues,
    }


def validate_dpo_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    reward_gaps: list[float] = []
    char_similarities: list[float] = []
    negative_types: Counter[str] = Counter()
    future_leaks = 0
    pair_keys: Counter[tuple[str, str, str]] = Counter()
    weights: list[float] = []

    for index, row in enumerate(rows):
        chosen = str(((row.get("chosen") or {}).get("content")) or "")
        rejected = str(((row.get("rejected") or {}).get("content")) or "")
        if not chosen or not rejected or normalize_text(chosen) == normalize_text(rejected):
            issues.append(f"dpo[{index}] chosen/rejected invalid")
            continue
        metadata = row.get("metadata", {})
        negative_type = str(metadata.get("negative_type") or "")
        negative_types[negative_type] += 1
        reward_gap = float(metadata.get("reward_gap") or 0.0)
        char_similarity_value = float(metadata.get("char_similarity") or 0.0)
        reward_gaps.append(reward_gap)
        char_similarities.append(char_similarity_value)
        weights.append(float(row.get("papo_weight") or 0.0))
        target_time = str(metadata.get("target_time") or row.get("target_time") or "")
        source_time = str(metadata.get("negative_source_time") or metadata.get("source_time") or "")
        if negative_type.startswith("same_user") and source_time and target_time and source_time >= target_time:
            future_leaks += 1
        pair_keys[(str(metadata.get("task_id") or ""), chosen, rejected)] += 1

    duplicate_pairs = sum(count - 1 for count in pair_keys.values() if count > 1)
    return {
        "rows": len(rows),
        "same_user_hard_negative_count": negative_types[SAME_USER_HARD_NEGATIVE],
        "context_hard_negative_count": negative_types[CONTEXT_HARD_NEGATIVE],
        "slot_mismatch_same_user_count": negative_types[SLOT_MISMATCH_SAME_USER],
        "negative_type_distribution": dict(negative_types),
        "reward_gap_distribution": summarize_numbers(reward_gaps),
        "papo_weight_distribution": summarize_numbers(weights),
        "average_char_similarity": _mean(char_similarities),
        "average_reward_gap": _mean(reward_gaps),
        "same_user_future_leak_ratio": future_leaks / len(rows) if rows else 0.0,
        "duplicate_pair_ratio": duplicate_pairs / len(rows) if rows else 0.0,
        "issues": issues,
        "passed": not issues,
    }


def validate_rerank_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    answer_counts = Counter()
    oracle_positions = Counter()
    for index, row in enumerate(rows):
        messages = row.get("messages") or []
        assistant = str(messages[-1].get("content") or "") if messages else ""
        answer_counts[assistant] += 1
        metadata = row.get("metadata", {})
        candidate_order = list(metadata.get("candidate_order") or [])
        if len(candidate_order) < 2:
            issues.append(f"rerank[{index}] candidate_count < 2")
            continue
        if assistant not in _candidate_letters(len(candidate_order)):
            issues.append(f"rerank[{index}] assistant answer not in candidate letters")
        if ORACLE_SOURCE not in candidate_order:
            issues.append(f"rerank[{index}] oracle missing from candidate_order")
        else:
            oracle_index = candidate_order.index(ORACLE_SOURCE)
            oracle_positions[_candidate_letters(len(candidate_order))[oracle_index]] += 1
        unique_sources = len(candidate_order)
        unique_texts = len(
            {
                normalize_text(text)
                for text in (
                    metadata.get("oracle_text", ""),
                    metadata.get("same_user_text", ""),
                    metadata.get("context_text", ""),
                )
                if normalize_text(text)
            }
        )
        if unique_sources < 2 or unique_texts < 2:
            issues.append(f"rerank[{index}] candidates are not sufficiently unique")

    shares = {
        letter: count / len(rows) if rows else 0.0
        for letter, count in sorted(answer_counts.items())
    }
    for letter, share in shares.items():
        if share > 0.45:
            issues.append(f"answer letter {letter} share exceeds 45%")
    return {
        "rows": len(rows),
        "answer_letter_distribution": dict(sorted(answer_counts.items())),
        "answer_letter_share": shares,
        "oracle_position_distribution": dict(sorted(oracle_positions.items())),
        "issues": issues,
        "passed": not issues,
    }


def validate_weighted_listwise_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    source_weights: dict[str, list[float]] = defaultdict(list)
    group_sums: dict[str, float] = defaultdict(float)
    group_weights: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        weight = float(row.get("papo_listwise_weight") or 0.0)
        metadata = row.get("metadata", {})
        group_id = str(metadata.get("group_id") or metadata.get("task_id") or f"row_{index}")
        source = str(metadata.get("candidate_source") or "")
        source_weights[source].append(weight)
        group_sums[group_id] += weight
        group_weights[group_id].append(weight)
        if not 0.0 < weight <= 1.0:
            issues.append(f"listwise[{index}] invalid weight")
    if rows and len({round(weight, 6) for weight in (row.get('papo_listwise_weight') for row in rows)}) == 1:
        issues.append("all papo_listwise_weight values are identical")
    for group_id, total in group_sums.items():
        if abs(total - 1.0) > 1e-6:
            issues.append(f"group {group_id} probability sum is {total:.6f}")
    return {
        "rows": len(rows),
        "groups": len(group_sums),
        "oracle_average_weight": _mean(source_weights[ORACLE_SOURCE]),
        "same_user_average_weight": _mean(source_weights[SAME_USER_SOURCE]),
        "context_average_weight": _mean(source_weights[CONTEXT_SOURCE]),
        "context_non_zero_weight_count": sum(1 for weight in source_weights[CONTEXT_SOURCE] if weight > 0.0),
        "group_probability_sum_distribution": summarize_numbers(list(group_sums.values())),
        "issues": issues,
        "passed": not issues,
    }


def build_examples_payload(
    sft_rows: list[dict[str, Any]],
    dpo_rows: list[dict[str, Any]],
    rerank_rows: list[dict[str, Any]],
    listwise_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "oracle_sft_examples": sft_rows[:3],
        "dpo_examples": dpo_rows[:3],
        "rerank_examples": rerank_rows[:3],
        "weighted_listwise_examples": listwise_rows[:6],
    }


def write_jsonl_dataset(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _normalize_wide_row(row: dict[str, Any]) -> dict[str, Any]:
    image_paths = _parse_image_paths(row.get("image_paths_json", ""))
    result = dict(row)
    result["image_paths"] = image_paths
    result["image_count"] = _int_or_zero(row.get("image_count"))
    result["history_count"] = _int_or_zero(row.get("history_count"))
    for field in [
        "oracle_margin_prob",
        "oracle_margin_reward",
        "oracle_prob",
        "oracle_reward_total",
        "same_user_prob",
        "same_user_reward_total",
        "same_user_semantic_similarity",
        "context_prob",
        "context_reward_total",
        "context_semantic_similarity",
    ]:
        result[field] = _float_or_zero(row.get(field))
    result["cross_user_analysis_count"] = _int_or_zero(row.get("cross_user_analysis_count"))
    result["dpo_rejected_count"] = _int_or_zero(row.get("dpo_rejected_count"))
    return result


def _parse_image_paths(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return [str(parsed)]


def _candidate_dpo_negatives(row: dict[str, Any], config: DPOExportConfig) -> list[dict[str, Any]]:
    negatives: list[dict[str, Any]] = []
    oracle_text = str(row.get("oracle_text") or "")
    target_time = str(row.get("target_time") or "")

    same_user_text = str(row.get("same_user_text") or "")
    same_user_char_similarity = char_similarity(oracle_text, same_user_text) if same_user_text else 0.0
    same_user_reward = float(row.get("same_user_reward_total") or 0.0)
    if (
        same_user_text
        and normalize_text(same_user_text) != normalize_text(oracle_text)
        and float(row.get("same_user_semantic_similarity") or 0.0) >= config.same_user_min_semantic_similarity
        and config.same_user_min_similarity <= same_user_char_similarity <= config.same_user_max_similarity
        and str(row.get("same_user_source_time") or "") < target_time
    ):
        negative_type = (
            SLOT_MISMATCH_SAME_USER
            if _is_slot_mismatch(oracle_text, same_user_text, same_user_char_similarity)
            else SAME_USER_HARD_NEGATIVE
        )
        negatives.append(
            _dpo_negative(
                row=row,
                text=same_user_text,
                negative_type=negative_type,
                negative_reward_total=same_user_reward,
                char_similarity_value=same_user_char_similarity,
                semantic_similarity=float(row.get("same_user_semantic_similarity") or 0.0),
                source_time=str(row.get("same_user_source_time") or ""),
                source_app=str(row.get("same_user_source_app") or ""),
            )
        )

    context_text = str(row.get("context_text") or "")
    context_char_similarity = char_similarity(oracle_text, context_text) if context_text else 0.0
    context_same_app = str(row.get("context_source_app") or "") == str(row.get("target_app") or "")
    if (
        context_text
        and normalize_text(context_text) != normalize_text(oracle_text)
        and config.context_min_similarity <= context_char_similarity <= config.context_max_similarity
        and (context_same_app or float(row.get("context_semantic_similarity") or 0.0) >= config.context_min_semantic_similarity)
    ):
        negatives.append(
            _dpo_negative(
                row=row,
                text=context_text,
                negative_type=CONTEXT_HARD_NEGATIVE,
                negative_reward_total=float(row.get("context_reward_total") or 0.0),
                char_similarity_value=context_char_similarity,
                semantic_similarity=float(row.get("context_semantic_similarity") or 0.0),
                source_time=str(row.get("context_source_time") or ""),
                source_app=str(row.get("context_source_app") or ""),
            )
        )

    negatives.sort(
        key=lambda item: (
            item["negative_type"] != SLOT_MISMATCH_SAME_USER,
            -item["reward_gap"],
            -item["char_similarity"],
        )
    )
    return negatives


def _dpo_negative(
    *,
    row: dict[str, Any],
    text: str,
    negative_type: str,
    negative_reward_total: float,
    char_similarity_value: float,
    semantic_similarity: float,
    source_time: str,
    source_app: str,
) -> dict[str, Any]:
    oracle_reward_total = float(row.get("oracle_reward_total") or 0.0)
    reward_gap = oracle_reward_total - negative_reward_total
    similarity_bonus = min(max(char_similarity_value, 0.0), 1.0)
    weight = 0.5 + 2.0 * reward_gap + 0.5 * similarity_bonus
    weight = min(max(weight, 0.5), 3.0)
    return {
        "text": text,
        "negative_type": negative_type,
        "negative_reward_total": negative_reward_total,
        "reward_gap": reward_gap,
        "char_similarity": char_similarity_value,
        "semantic_similarity": semantic_similarity,
        "weight": weight,
        "source_time": source_time,
        "source_app": source_app,
    }


def _is_slot_mismatch(oracle_text: str, same_user_text: str, similarity_value: float) -> bool:
    if not (0.70 <= similarity_value < 1.0):
        return False
    oracle_tail = _tail_chinese(oracle_text)
    same_user_tail = _tail_chinese(same_user_text)
    return bool(oracle_tail and same_user_tail and oracle_tail != same_user_tail)


def _tail_chinese(text: str) -> str:
    chars = re.findall(r"[\u4e00-\u9fff]", str(text or ""))
    if not chars:
        return ""
    for length in range(8, 1, -1):
        if len(chars) >= length:
            return "".join(chars[-length:])
    return "".join(chars)


def _build_rerank_candidates(row: dict[str, Any]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for source, field in [
        (ORACLE_SOURCE, "oracle_text"),
        (SAME_USER_SOURCE, "same_user_text"),
        (CONTEXT_SOURCE, "context_text"),
    ]:
        text = str(row.get(field) or "").strip()
        normalized = normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append({"source": source, "text": text})
    return candidates


def _build_weighted_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, field, reward_field in [
        (ORACLE_SOURCE, "oracle_text", "oracle_reward_total"),
        (SAME_USER_SOURCE, "same_user_text", "same_user_reward_total"),
        (CONTEXT_SOURCE, "context_text", "context_reward_total"),
    ]:
        text = str(row.get(field) or "").strip()
        normalized = normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(
            {
                "source": source,
                "text": text,
                "reward": float(row.get(reward_field) or 0.0),
            }
        )
    return candidates


def _dynamic_candidate_probabilities(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    config: WeightedListwiseExportConfig,
) -> dict[str, float]:
    rewards = {candidate["source"]: float(candidate["reward"]) for candidate in candidates}
    probs = _softmax_distribution(rewards, config.temperature)

    same_user_probability = min(probs.get(SAME_USER_SOURCE, 0.0), config.max_same_user_prob)
    context_probability = min(probs.get(CONTEXT_SOURCE, 0.0), config.max_context_prob)
    context_hard = _context_is_hard(row, config.context_hardness_threshold)
    if CONTEXT_SOURCE in rewards and context_hard:
        context_probability = max(context_probability, config.min_context_prob)

    other_total = same_user_probability + context_probability
    oracle_probability = 1.0 - other_total
    if oracle_probability < config.min_oracle_prob:
        cap_total = max(0.0, 1.0 - config.min_oracle_prob)
        scale = cap_total / other_total if other_total > 0 else 0.0
        same_user_probability *= scale
        context_probability *= scale
        oracle_probability = config.min_oracle_prob
    if oracle_probability > config.max_oracle_prob:
        needed = oracle_probability - config.max_oracle_prob
        slack_same = max(0.0, config.max_same_user_prob - same_user_probability) if SAME_USER_SOURCE in rewards else 0.0
        slack_context = max(0.0, config.max_context_prob - context_probability) if CONTEXT_SOURCE in rewards else 0.0
        slack_total = slack_same + slack_context
        if slack_total > 0:
            if slack_same > 0:
                same_user_probability += needed * (slack_same / slack_total)
                same_user_probability = min(same_user_probability, config.max_same_user_prob)
            if slack_context > 0:
                context_probability += needed * (slack_context / slack_total)
                context_probability = min(context_probability, config.max_context_prob)
        oracle_probability = 1.0 - same_user_probability - context_probability

    probs_out = {
        ORACLE_SOURCE: max(0.0, oracle_probability),
        SAME_USER_SOURCE: max(0.0, same_user_probability) if SAME_USER_SOURCE in rewards else 0.0,
        CONTEXT_SOURCE: max(0.0, context_probability) if CONTEXT_SOURCE in rewards else 0.0,
    }
    total = sum(probs_out[source] for source in rewards)
    if total <= 0.0:
        return {source: 1.0 / len(rewards) for source in rewards}
    return {source: probs_out[source] / total for source in rewards}


def _context_is_hard(row: dict[str, Any], threshold: float) -> bool:
    oracle_text = str(row.get("oracle_text") or "")
    context_text = str(row.get("context_text") or "")
    if not oracle_text or not context_text:
        return False
    if normalize_text(oracle_text) == normalize_text(context_text):
        return False
    char_value = char_similarity(oracle_text, context_text)
    semantic = float(row.get("context_semantic_similarity") or 0.0)
    same_app = str(row.get("context_source_app") or "") == str(row.get("target_app") or "")
    return char_value >= threshold and (same_app or semantic >= threshold)


def char_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, str(left or ""), str(right or "")).ratio()


def summarize_numbers(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "min": None, "max": None, "p50": None, "p90": None, "p95": None}
    ordered = sorted(finite)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p50": _percentile(ordered, 50),
        "p90": _percentile(ordered, 90),
        "p95": _percentile(ordered, 95),
    }


def _softmax_distribution(rewards: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    values = {key: math.exp(value / temperature) for key, value in rewards.items()}
    total = sum(values.values())
    if total <= 0.0:
        return {key: 1.0 / len(values) for key in values}
    return {key: value / total for key, value in values.items()}


def _candidate_letters(count: int) -> list[str]:
    letters = ["A", "B", "C", "D", "E", "F"]
    if count > len(letters):
        raise ValueError(f"unsupported candidate count: {count}")
    return letters[:count]


def _stable_seed(task_id: str, seed: int) -> int:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return seed + int(digest[:12], 16) % 1_000_000


def _int_or_zero(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _ratio(items: list[dict[str, Any]], predicate: Any) -> float:
    if not items:
        return 0.0
    hits = 0
    for item in items:
        if predicate(item):
            hits += 1
    return hits / len(items)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
