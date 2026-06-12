from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_prediction import merge_prediction_shards, read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and validate resumable Proactive prediction shards.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args()

    report = merge_prediction_shards(
        read_jsonl(args.tasks),
        args.shards,
        args.output,
        task_path=args.tasks,
        adapter_dir=args.adapter,
        allow_errors=args.allow_errors,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("PROACTIVE PREDICTION MERGE VALIDATION PASSED")


if __name__ == "__main__":
    main()
