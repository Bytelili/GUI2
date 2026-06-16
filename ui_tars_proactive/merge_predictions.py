from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.data_protocol import sha256_file  # noqa: E402
from papo.proactive_prediction import RESULT_FIELDS, read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge generic UI-TARS Proactive prediction shards.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args()

    tasks = read_jsonl(args.tasks)
    task_ids = [str(task.get("task_id") or "") for task in tasks]
    if any(not task_id for task_id in task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("Prediction task file contains empty or duplicate task IDs")
    records: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for shard in args.shards:
        for row in read_jsonl(shard):
            task_id = str(row.get("task_id") or "")
            if task_id in records:
                duplicates.add(task_id)
            records[task_id] = row
    unknown = set(records) - set(task_ids)
    missing = set(task_ids) - set(records)
    errors = [
        task_id
        for task_id, row in records.items()
        if str(row.get("error") or "") or str(row.get("predicted_intent") or "").strip().upper() == "ERROR"
    ]
    if duplicates or unknown or missing or (errors and not args.allow_errors):
        raise ValueError(
            "Prediction merge validation failed: "
            f"duplicates={len(duplicates)}, unknown={len(unknown)}, missing={len(missing)}, errors={len(errors)}"
        )
    ordered = [records[task_id] for task_id in task_ids]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in RESULT_FIELDS} for row in ordered)
    report = {
        "status": "passed",
        "model_label": args.model_label,
        "task_path": str(Path(args.tasks).resolve()),
        "task_sha256": sha256_file(args.tasks),
        "output_csv": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "records": len(ordered),
        "errors": len(errors),
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("UI-TARS PROACTIVE PREDICTION MERGE PASSED")


if __name__ == "__main__":
    main()
