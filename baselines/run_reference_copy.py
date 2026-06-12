from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reference-copy personalized execution baseline.")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variants", default="", help="Comma-separated variants; empty means all.")
    parser.add_argument(
        "--fallback",
        choices=["wait", "finished", "last"],
        default="wait",
        help="Prediction when the reference trajectory has no action at the current step.",
    )
    args = parser.parse_args()

    rows = _read_jsonl(args.prompts)
    selected_variants = {item.strip() for item in args.variants.split(",") if item.strip()}
    if selected_variants:
        rows = [row for row in rows if row.get("variant") in selected_variants]
    predictions = []
    for row in rows:
        references = list(row.get("reference_actions") or [])
        step_index = int(row.get("step_index", 0) or 0)
        if step_index < len(references):
            prediction = references[step_index]
        elif args.fallback == "last" and references:
            prediction = references[-1]
        else:
            prediction = f"{args.fallback}()"
        predictions.append({**row, "prediction": prediction, "prediction_source": "reference_copy"})
    write_jsonl(args.out, predictions)
    print(f"predictions: {len(predictions)}")
    print(f"wrote: {args.out}")


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


if __name__ == "__main__":
    main()
