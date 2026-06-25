from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.llamafactory_export import dataset_info  # noqa: E402
from papo.proactive_fixed_export import read_jsonish_rows  # noqa: E402


EXPECTED_DATASETS = [
    "papo_proactive_oracle_sft_train",
    "papo_proactive_oracle_sft_eval",
    "papo_proactive_dpo_train",
    "papo_proactive_dpo_eval",
    "papo_proactive_rerank_train",
    "papo_proactive_rerank_eval",
    "papo_proactive_weighted_listwise_train",
    "papo_proactive_weighted_listwise_eval",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test proactive_fixed dataset_info registration.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=8)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    base_dir = data_dir.parent
    info = dataset_info()
    report: dict[str, Any] = {
        "status": "passed",
        "data_dir": str(data_dir),
        "datasets": {},
    }
    failures: list[str] = []

    for name in EXPECTED_DATASETS:
        if name not in info:
            failures.append(f"missing_dataset::{name}")
            continue
        record = info[name]
        file_name = str(record.get("file_name") or "")
        path = base_dir / file_name
        if not path.exists():
            failures.append(f"missing_file::{name}")
            continue
        rows = read_jsonish_rows(path)[: args.max_samples]
        report["datasets"][name] = {
            "file_name": file_name,
            "columns": record.get("columns"),
            "tags": record.get("tags"),
            "rows_checked": len(rows),
            "preview": rows[0] if rows else None,
        }
        if not rows:
            failures.append(f"empty_dataset::{name}")

    if failures:
        report["status"] = "failed"
        report["failures"] = failures

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
