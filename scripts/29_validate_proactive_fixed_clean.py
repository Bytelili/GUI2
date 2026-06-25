from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.io import write_json  # noqa: E402
from papo.proactive_fixed_export import (  # noqa: E402
    read_jsonish_rows,
    validate_dpo_rows,
    validate_rerank_rows,
    validate_sft_rows,
    validate_weighted_listwise_rows,
)


FILE_SPECS = {
    "proactive_oracle_sft_train.jsonl": "sft",
    "proactive_oracle_sft_eval.jsonl": "sft",
    "proactive_dpo_train.jsonl": "dpo",
    "proactive_dpo_eval.jsonl": "dpo",
    "proactive_rerank_train.jsonl": "rerank",
    "proactive_rerank_eval.jsonl": "rerank",
    "proactive_weighted_listwise_train.jsonl": "listwise",
    "proactive_weighted_listwise_eval.jsonl": "listwise",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate proactive_fixed_clean datasets.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--project_root", default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    project_root = Path(args.project_root) if args.project_root else None

    report: dict[str, Any] = {
        "status": "passed",
        "data_dir": str(data_dir),
        "files": {},
        "summary": {},
    }
    failures: list[str] = []
    image_style_counter: Counter[str] = Counter()

    for filename, kind in FILE_SPECS.items():
        path = data_dir / filename
        if not path.exists():
            failures.append(f"missing_file::{filename}")
            continue
        rows = read_jsonish_rows(path)
        prompt_report = _validate_prompt_cleanliness(rows, kind)
        image_report = _validate_images(rows, project_root)
        image_style_counter.update(image_report["path_style_distribution"])
        if kind == "sft":
            core = validate_sft_rows(rows)
        elif kind == "dpo":
            core = validate_dpo_rows(rows)
        elif kind == "rerank":
            core = validate_rerank_rows(rows)
        else:
            core = validate_weighted_listwise_rows(rows)

        report["files"][filename] = {
            "rows": len(rows),
            "core": core,
            "prompt": prompt_report,
            "images": image_report,
        }
        if not core.get("passed", False):
            failures.append(f"{filename}::core")
        if not prompt_report.get("passed", False):
            failures.append(f"{filename}::prompt")
        if not image_report.get("passed", False):
            failures.append(f"{filename}::images")

    report["summary"] = {
        "file_count": len(report["files"]),
        "path_style_distribution": dict(image_style_counter),
    }
    if failures:
        report["status"] = "failed"
        report["failures"] = failures

    write_json(out_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


def _validate_prompt_cleanliness(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    bad_tags = 0
    missing_intent_phrase = 0
    repeated_system = 0
    for row in rows:
        prompt = _row_prompt(row, kind)
        if "[system]" in prompt.lower() or "[user]" in prompt.lower():
            bad_tags += 1
        if "Infer the user's current intent" not in prompt:
            missing_intent_phrase += 1
        if prompt.count("You are a personalized Android GUI agent") > 1:
            repeated_system += 1
    passed = bad_tags == 0 and missing_intent_phrase == 0 and repeated_system == 0
    return {
        "passed": passed,
        "tagged_prompt_count": bad_tags,
        "missing_intent_phrase_count": missing_intent_phrase,
        "repeated_system_prompt_count": repeated_system,
    }


def _validate_images(rows: list[dict[str, Any]], project_root: Path | None) -> dict[str, Any]:
    issues: list[str] = []
    style_counter: Counter[str] = Counter()
    missing_local = 0
    for index, row in enumerate(rows):
        images = row.get("images")
        if not isinstance(images, list):
            issues.append(f"row[{index}] images is not a list")
            continue
        for image in images:
            image_text = str(image or "")
            if not image_text:
                issues.append(f"row[{index}] empty image path")
                continue
            style_counter[_path_style(image_text)] += 1
            if project_root is not None and _path_style(image_text) == "relative":
                if not (project_root / image_text).exists():
                    missing_local += 1
    if style_counter.get("absolute", 0) > 0:
        issues.append("absolute image paths remain")
    return {
        "passed": not issues,
        "path_style_distribution": dict(style_counter),
        "missing_local_files": missing_local,
        "issues": issues,
    }


def _row_prompt(row: dict[str, Any], kind: str) -> str:
    key = "conversations" if kind == "dpo" else "messages"
    messages = row.get(key) or []
    if len(messages) < 2:
        return ""
    return str(messages[1].get("value") or messages[1].get("content") or "")


def _path_style(path: str) -> str:
    if path.startswith("/") or (len(path) > 2 and path[1] == ":" and path[2] in ("\\", "/")):
        return "absolute"
    return "relative"


if __name__ == "__main__":
    main()
