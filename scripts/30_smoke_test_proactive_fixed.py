from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_fixed_export import read_jsonish_rows  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Smoke test proactive_fixed_clean datasets.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=16)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    report: dict[str, Any] = {"status": "passed", "dataset_dir": str(dataset_dir), "files": {}}
    failures: list[str] = []

    for filename, kind in FILE_SPECS.items():
        path = dataset_dir / filename
        if not path.exists():
            failures.append(f"missing_file::{filename}")
            continue
        rows = read_jsonish_rows(path)[: args.max_samples]
        summary = _summarize_rows(rows, kind)
        report["files"][filename] = summary
        if not summary["passed"]:
            failures.append(filename)

    if failures:
        report["status"] = "failed"
        report["failures"] = failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


def _summarize_rows(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    issues: list[str] = []
    example = rows[0] if rows else {}
    image_counts = [len(row.get("images", [])) for row in rows if isinstance(row.get("images"), list)]

    for index, row in enumerate(rows):
        if kind == "dpo":
            if not row.get("chosen") or not row.get("rejected"):
                issues.append(f"row[{index}] missing chosen/rejected")
                continue
            if str((row.get("chosen") or {}).get("value") or "").strip() == str(
                (row.get("rejected") or {}).get("value") or ""
            ).strip():
                issues.append(f"row[{index}] chosen equals rejected")
        else:
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                issues.append(f"row[{index}] missing message triplet")
                continue
            if kind == "rerank":
                answer = str(messages[-1].get("value") or "")
                if answer not in {"A", "B", "C", "D"}:
                    issues.append(f"row[{index}] rerank answer is not A/B/C/D")

    return {
        "passed": not issues,
        "rows_checked": len(rows),
        "image_count_min": min(image_counts) if image_counts else 0,
        "image_count_max": max(image_counts) if image_counts else 0,
        "example": example,
        "issues": issues,
    }


if __name__ == "__main__":
    main()
