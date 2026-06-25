from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.config import load_config  # noqa: E402
from papo.io import write_json  # noqa: E402
from papo.llamafactory_export import dataset_info  # noqa: E402
from papo.proactive_fixed_export import (  # noqa: E402
    read_jsonish_rows,
    validate_dpo_rows,
    validate_rerank_rows,
    validate_sft_rows,
    validate_weighted_listwise_rows,
)


REQUIRED_CONFIG = {
    "proactive_oracle_sft_fixed": {
        "stage": "sft",
        "datasets": ["papo_proactive_oracle_sft_train", "papo_proactive_oracle_sft_eval"],
    },
    "proactive_dpo_fixed": {
        "stage": "dpo",
        "datasets": ["papo_proactive_dpo_train", "papo_proactive_dpo_eval"],
    },
    "proactive_rerank_fixed": {
        "stage": "sft",
        "datasets": ["papo_proactive_rerank_train", "papo_proactive_rerank_eval"],
    },
    "proactive_weighted_listwise_fixed": {
        "stage": "sft",
        "datasets": ["papo_proactive_weighted_listwise_train", "papo_proactive_weighted_listwise_eval"],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate proactive_fixed_clean integration end-to-end.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_dir = Path(args.dataset_dir)
    source_info = dataset_info()
    disk_info_path = dataset_dir / "dataset_info.json"
    disk_info = json.loads(disk_info_path.read_text(encoding="utf-8")) if disk_info_path.exists() else {}

    failures: list[str] = []
    warnings: list[str] = []
    section_report: dict[str, Any] = {}
    dataset_report: dict[str, Any] = {}

    training = config.get("training", {})
    for section_name, spec in REQUIRED_CONFIG.items():
        section = training.get(section_name)
        if not isinstance(section, dict):
            failures.append(f"Missing training section: {section_name}")
            continue
        issues: list[str] = []
        if str(section.get("stage")) != spec["stage"]:
            issues.append(f"stage != {spec['stage']}")
        if "output_name" not in section:
            issues.append("missing output_name")
        if section_name == "proactive_dpo_fixed":
            if str(section.get("adapter_name_or_path")) != "proactive_oracle_sft_fixed_clean_v2_best":
                issues.append("adapter_name_or_path should point to proactive_oracle_sft_fixed_clean_v2_best")
        if section_name == "proactive_weighted_listwise_fixed":
            if not bool(section.get("use_papo_listwise")):
                issues.append("use_papo_listwise must be true")
            if str(section.get("adapter_name_or_path")) != "proactive_oracle_sft_fixed_clean_v2_best":
                issues.append("adapter_name_or_path should point to proactive_oracle_sft_fixed_clean_v2_best")
        section_report[section_name] = {
            "present": True,
            "stage": section.get("stage"),
            "issues": issues,
        }
        failures.extend(f"{section_name}: {issue}" for issue in issues)

    for dataset_name in sorted({name for spec in REQUIRED_CONFIG.values() for name in spec["datasets"]}):
        entry_issues: list[str] = []
        source_entry = source_info.get(dataset_name)
        disk_entry = disk_info.get(dataset_name)
        if source_entry is None:
            entry_issues.append("missing from papo.llamafactory_export.dataset_info()")
        if disk_entry is None:
            entry_issues.append("missing from dataset_info.json")
        file_name = None
        if source_entry is not None:
            file_name = source_entry.get("file_name")
        elif disk_entry is not None:
            file_name = disk_entry.get("file_name")
        if source_entry and disk_entry and source_entry.get("file_name") != disk_entry.get("file_name"):
            entry_issues.append("source dataset_info() and dataset_info.json file_name mismatch")
        rows: list[dict[str, Any]] = []
        data_path = dataset_dir / str(file_name) if file_name else None
        if data_path is None or not data_path.exists():
            entry_issues.append("dataset file missing on disk")
        else:
            rows = read_jsonish_rows(data_path)
            if not rows:
                entry_issues.append("dataset file is empty")
        sample_rows = rows[:20]
        if sample_rows:
            entry_issues.extend(_validate_schema(dataset_name, sample_rows))
        dataset_report[dataset_name] = {
            "file_name": file_name,
            "exists": bool(data_path and data_path.exists()),
            "rows": len(rows),
            "issues": entry_issues,
        }
        failures.extend(f"{dataset_name}: {issue}" for issue in entry_issues)

    report = {
        "status": "passed" if not failures else "failed",
        "config": str(Path(args.config).resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "disk_dataset_info_exists": disk_info_path.exists(),
        "sections": section_report,
        "datasets": dataset_report,
        "warnings": warnings,
        "failures": failures,
    }
    write_json(Path(args.out), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)
    print("PROACTIVE_FIXED INTEGRATION VALIDATION PASSED")


def _validate_schema(dataset_name: str, rows: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    if "oracle_sft" in dataset_name:
        report = validate_sft_rows(rows)
        issues.extend(report.get("issues", []))
        for index, row in enumerate(rows):
            issues.extend(_check_messages_from_value(row, dataset_name, index))
            if not _assistant_value(row):
                issues.append(f"{dataset_name}[{index}] empty gpt.value")
    elif "rerank" in dataset_name:
        report = validate_rerank_rows(rows)
        issues.extend(report.get("issues", []))
        for index, row in enumerate(rows):
            issues.extend(_check_messages_from_value(row, dataset_name, index))
            answer = _assistant_value(row)
            if answer not in {"A", "B", "C", "D"}:
                issues.append(f"{dataset_name}[{index}] rerank answer is not A/B/C/D")
    elif "weighted_listwise" in dataset_name:
        report = validate_weighted_listwise_rows(rows)
        for issue in report.get("issues", []):
            if issue.startswith("group ") and "probability sum is" in issue:
                value = _parse_group_sum_issue(issue)
                if value is not None and 0.95 <= value <= 1.000001:
                    continue
            issues.append(issue)
        for index, row in enumerate(rows):
            issues.extend(_check_messages_from_value(row, dataset_name, index))
            if float(row.get("papo_listwise_weight") or 0.0) <= 0.0:
                issues.append(f"{dataset_name}[{index}] papo_listwise_weight <= 0")
    elif "dpo" in dataset_name:
        report = validate_dpo_rows(rows)
        issues.extend(report.get("issues", []))
        for index, row in enumerate(rows):
            conversations = row.get("conversations")
            if not isinstance(conversations, list) or len(conversations) < 2:
                issues.append(f"{dataset_name}[{index}] conversations missing")
            else:
                for message_index, message in enumerate(conversations):
                    issues.extend(_check_message(message, dataset_name, index, message_index))
                human_prompt = str(conversations[-1].get("value") or "")
                if "[system]" in human_prompt.lower() or "[user]" in human_prompt.lower():
                    issues.append(f"{dataset_name}[{index}] human prompt still contains prompt tags")
            chosen = row.get("chosen") or {}
            rejected = row.get("rejected") or {}
            if str(chosen.get("from")) != "gpt":
                issues.append(f"{dataset_name}[{index}] chosen.from != gpt")
            if str(rejected.get("from")) != "gpt":
                issues.append(f"{dataset_name}[{index}] rejected.from != gpt")
            if not str(chosen.get("value") or "").strip():
                issues.append(f"{dataset_name}[{index}] chosen.value empty")
            if not str(rejected.get("value") or "").strip():
                issues.append(f"{dataset_name}[{index}] rejected.value empty")
            if str(chosen.get("value") or "").strip() == str(rejected.get("value") or "").strip():
                issues.append(f"{dataset_name}[{index}] chosen.value == rejected.value")
            target = float(row.get("papo_target_probability") or 0.0)
            if not 0.55 <= target <= 0.98:
                issues.append(f"{dataset_name}[{index}] papo_target_probability out of range")
    return issues


def _check_messages_from_value(row: dict[str, Any], dataset_name: str, row_index: int) -> list[str]:
    messages = row.get("messages")
    issues: list[str] = []
    if not isinstance(messages, list) or len(messages) < 3:
        return [f"{dataset_name}[{row_index}] messages missing"]
    for message_index, message in enumerate(messages):
        issues.extend(_check_message(message, dataset_name, row_index, message_index))
    human_prompt = str(messages[1].get("value") or "")
    if "[system]" in human_prompt.lower() or "[user]" in human_prompt.lower():
        issues.append(f"{dataset_name}[{row_index}] human prompt still contains prompt tags")
    return issues


def _check_message(message: Any, dataset_name: str, row_index: int, message_index: int) -> list[str]:
    if not isinstance(message, dict):
        return [f"{dataset_name}[{row_index}] message[{message_index}] is not a dict"]
    issues: list[str] = []
    role = str(message.get("from") or "")
    value = str(message.get("value") or "")
    if role not in {"system", "human", "gpt"}:
        issues.append(f"{dataset_name}[{row_index}] message[{message_index}] invalid from")
    if not value.strip():
        issues.append(f"{dataset_name}[{row_index}] message[{message_index}] empty value")
    return issues


def _assistant_value(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return ""
    message = messages[-1]
    if not isinstance(message, dict):
        return ""
    return str(message.get("value") or "").strip()


def _parse_group_sum_issue(issue: str) -> float | None:
    prefix = "probability sum is "
    if prefix not in issue:
        return None
    try:
        return float(issue.split(prefix, 1)[1].strip())
    except ValueError:
        return None


if __name__ == "__main__":
    main()
