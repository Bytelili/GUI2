from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and validate Proactive sampled-candidate shards.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    task_ids = [str(row.get("task_id") or "") for row in _read_jsonl(Path(args.tasks))]
    if any(not task_id for task_id in task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError("Task file contains empty or duplicate task IDs")
    records: dict[str, dict] = {}
    duplicates: set[str] = set()
    for shard in args.shards:
        for row in _read_jsonl(Path(shard)):
            task_id = str(row.get("task_id") or "")
            if task_id in records:
                duplicates.add(task_id)
            records[task_id] = row
    missing = set(task_ids) - set(records)
    unknown = set(records) - set(task_ids)
    if duplicates or missing or unknown:
        raise ValueError(
            f"Candidate shard merge failed: duplicates={len(duplicates)}, "
            f"missing={len(missing)}, unknown={len(unknown)}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for task_id in task_ids:
            file.write(json.dumps(records[task_id], ensure_ascii=False) + "\n")
    print(f"Merged {len(task_ids)} candidate records -> {output}")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
