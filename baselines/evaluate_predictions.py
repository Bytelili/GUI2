from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common import write_json
from baselines.metrics import prediction_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate teacher-forced FingerTip baseline predictions.")
    parser.add_argument("--predictions", required=True, help="JSONL rows containing prediction and target_action.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = _read_jsonl(args.predictions)
    variants = sorted({str(row.get("variant") or "unknown") for row in rows})
    report: dict[str, Any] = {
        "overall": prediction_report(rows),
        "variants": {
            variant: prediction_report([row for row in rows if str(row.get("variant") or "unknown") == variant])
            for variant in variants
        },
    }
    write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


if __name__ == "__main__":
    main()
