from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.io import write_json  # noqa: E402
from papo.proactive_fixed_export import (  # noqa: E402
    char_similarity,
    read_jsonish_rows,
    summarize_numbers,
)


FILE_NAMES = {
    "proactive_dpo_train.jsonl": "dpo",
    "proactive_dpo_eval.jsonl": "dpo",
    "proactive_oracle_sft_train.jsonl": "sft",
    "proactive_oracle_sft_eval.jsonl": "sft",
    "proactive_rerank_train.jsonl": "rerank",
    "proactive_rerank_eval.jsonl": "rerank",
    "proactive_weighted_listwise_train.jsonl": "listwise",
    "proactive_weighted_listwise_eval.jsonl": "listwise",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate proactive_fixed_clean datasets.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.out)
    report: dict[str, Any] = {
        "status": "passed",
        "data_dir": str(data_dir),
        "files": {},
        "summary": {},
    }
    failures: list[str] = []
    warnings: list[str] = []

    for filename, kind in FILE_NAMES.items():
        path = data_dir / filename
        if not path.exists():
            failures.append(f"missing::{filename}")
            continue
        rows = read_jsonish_rows(path)
        file_report = validate_rows(rows, kind)
        file_report["rows"] = len(rows)
        report["files"][filename] = file_report
        if not file_report["passed"]:
            failures.append(filename)
        warnings.extend(f"{filename}::{item}" for item in file_report.get("warnings", []))

    report["summary"] = {
        "file_count": len(report["files"]),
        "warnings": warnings,
    }
    if failures:
        report["status"] = "failed"
        report["failures"] = failures
    elif warnings:
        report["status"] = "warning"

    write_json(out_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


def validate_rows(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    if kind == "sft":
        return validate_sft_like(rows, rerank=False, listwise=False)
    if kind == "rerank":
        return validate_sft_like(rows, rerank=True, listwise=False)
    if kind == "listwise":
        return validate_sft_like(rows, rerank=False, listwise=True)
    return validate_dpo(rows)


def validate_sft_like(rows: list[dict[str, Any]], *, rerank: bool, listwise: bool) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    from_counter: Counter[str] = Counter()
    answer_counter: Counter[str] = Counter()
    candidate_source_counter: Counter[str] = Counter()
    weights_by_source: dict[str, list[float]] = defaultdict(list)
    group_sums: dict[str, float] = defaultdict(float)

    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list):
            issues.append(f"row[{index}] messages is not a list")
            continue
        if len(messages) < 3:
            issues.append(f"row[{index}] missing system/human/gpt triplet")
            continue
        roles = []
        for message in messages:
            if not isinstance(message, dict):
                issues.append(f"row[{index}] message is not an object")
                continue
            if "from" not in message or "value" not in message:
                issues.append(f"row[{index}] message missing from/value")
                continue
            role = str(message.get("from") or "")
            roles.append(role)
            from_counter[role] += 1
            if role not in {"system", "human", "gpt"}:
                issues.append(f"row[{index}] invalid from role: {role}")
        if "system" not in roles:
            issues.append(f"row[{index}] missing system message")
        if "human" not in roles:
            issues.append(f"row[{index}] missing human message")
        if "gpt" not in roles:
            issues.append(f"row[{index}] missing gpt message")

        human_value = message_value(messages, "human")
        gpt_value = message_value(messages, "gpt")
        if "[system]" in human_value.lower() or "[user]" in human_value.lower():
            issues.append(f"row[{index}] dirty prompt tags remain")
        if not gpt_value.strip():
            issues.append(f"row[{index}] empty gpt.value")

        if rerank:
            answer_counter[gpt_value] += 1
            if gpt_value not in {"A", "B", "C", "D"}:
                issues.append(f"row[{index}] rerank answer is not A/B/C/D")
            if "Candidates" not in human_value:
                issues.append(f"row[{index}] rerank prompt missing Candidates section")
            if "A." not in human_value or "B." not in human_value:
                issues.append(f"row[{index}] rerank prompt has fewer than two visible candidates")

        if listwise:
            if "papo_listwise_weight" not in row:
                issues.append(f"row[{index}] missing papo_listwise_weight")
                continue
            weight = float(row.get("papo_listwise_weight") or 0.0)
            if weight <= 0.0:
                issues.append(f"row[{index}] non-positive papo_listwise_weight")
            source = str((row.get("metadata") or {}).get("candidate_source") or "")
            candidate_source_counter[source] += 1
            weights_by_source[source].append(weight)
            group_id = str((row.get("metadata") or {}).get("group_id") or (row.get("metadata") or {}).get("task_id") or f"row_{index}")
            group_sums[group_id] += weight

    if rerank and rows:
        for label, count in sorted(answer_counter.items()):
            if count / len(rows) > 0.5:
                warnings.append(f"rerank answer {label} share exceeds 50%")

    if listwise:
        unique_weights = {round(float(row.get("papo_listwise_weight") or 0.0), 6) for row in rows}
        if len(unique_weights) <= 1:
            issues.append("papo_listwise_weight is globally fixed")
        context_mean = average(weights_by_source.get("context", []))
        if context_mean is not None and abs(context_mean) <= 1e-12:
            issues.append("context average weight is zero")
        for group_id, total in group_sums.items():
            if abs(total - 1.0) > 1e-5:
                issues.append(f"group {group_id} weight sum is {total:.6f}")

    report: dict[str, Any] = {
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "from_distribution": dict(sorted(from_counter.items())),
    }
    if rerank:
        report["answer_distribution"] = dict(sorted(answer_counter.items()))
    if listwise:
        report["candidate_source_distribution"] = dict(sorted(candidate_source_counter.items()))
        report["oracle_average_weight"] = average(weights_by_source.get("oracle", []))
        report["same_user_average_weight"] = average(weights_by_source.get("same_user", []))
        report["context_average_weight"] = average(weights_by_source.get("context", []))
        report["group_weight_sum_distribution"] = summarize_numbers(list(group_sums.values()))
    return report


def validate_dpo(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    negative_types: Counter[str] = Counter()
    reward_gaps: list[float] = []
    target_probabilities: list[float] = []
    weights: list[float] = []
    char_sims: list[float] = []

    for index, row in enumerate(rows):
        conversations = row.get("conversations")
        if not isinstance(conversations, list):
            issues.append(f"row[{index}] conversations is not a list")
            continue
        for item in conversations:
            if not isinstance(item, dict) or "from" not in item or "value" not in item:
                issues.append(f"row[{index}] invalid conversations schema")
                break

        chosen = row.get("chosen") or {}
        rejected = row.get("rejected") or {}
        if "from" not in chosen or "value" not in chosen:
            issues.append(f"row[{index}] chosen missing from/value")
            continue
        if "from" not in rejected or "value" not in rejected:
            issues.append(f"row[{index}] rejected missing from/value")
            continue
        if str(chosen.get("from")) != "gpt":
            issues.append(f"row[{index}] chosen.from is not gpt")
        if str(rejected.get("from")) != "gpt":
            issues.append(f"row[{index}] rejected.from is not gpt")
        chosen_value = str(chosen.get("value") or "")
        rejected_value = str(rejected.get("value") or "")
        if chosen_value == rejected_value:
            issues.append(f"row[{index}] chosen.value equals rejected.value")

        if "papo_weight" not in row:
            issues.append(f"row[{index}] missing papo_weight")
            continue
        if "papo_target_probability" not in row:
            issues.append(f"row[{index}] missing papo_target_probability")
            continue
        weight = float(row.get("papo_weight") or 0.0)
        target = float(row.get("papo_target_probability") or 0.0)
        if not 0.5 <= weight <= 3.0:
            issues.append(f"row[{index}] papo_weight out of range")
        if not 0.55 <= target <= 0.98:
            issues.append(f"row[{index}] papo_target_probability out of range")

        metadata = row.get("metadata") or {}
        negative_types[str(metadata.get("negative_type") or "")] += 1
        reward_gap = float(metadata.get("reward_gap") or 0.0)
        reward_gaps.append(reward_gap)
        target_probabilities.append(target)
        weights.append(weight)
        char_sims.append(char_similarity(chosen_value, rejected_value))

    unique_targets = {round(value, 6) for value in target_probabilities if math.isfinite(value)}
    if target_probabilities and len(unique_targets) == 1:
        issues.append("papo_target_probability has a single unique value")

    return {
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "negative_type_distribution": dict(sorted(negative_types.items())),
        "reward_gap_distribution": summarize_numbers(reward_gaps),
        "papo_weight_distribution": summarize_numbers(weights),
        "papo_target_probability_distribution": summarize_numbers(target_probabilities),
        "chosen_rejected_char_similarity_distribution": summarize_numbers(char_sims),
    }


def message_value(messages: list[dict[str, Any]], role: str) -> str:
    for message in messages:
        if str(message.get("from") or "") == role:
            return str(message.get("value") or "")
    return ""


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
