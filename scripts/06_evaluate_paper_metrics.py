from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.io import read_jsonl, write_json  # noqa: E402
from papo.paper_metrics import (  # noqa: E402
    execution_metrics,
    execution_reference_metrics,
    papo_tree_proxy_metrics,
    proactive_metrics,
    suggestion_task_readiness,
)


def read_rows(path: str) -> list[dict[str, Any]]:
    if not path:
        return []
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        return read_jsonl(input_path)
    if input_path.suffix.lower() == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FingerTip paper metrics and PAPO offline proxies.")
    parser.add_argument("--suggestion_results", default="")
    parser.add_argument("--execution_results", default="")
    parser.add_argument("--trees", default="")
    parser.add_argument("--suggestion_tasks", default="")
    parser.add_argument("--execution_tasks", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "metric_source": "FingerTip 20K, arXiv:2507.21071, Section 5.1",
        "proactive_suggestion": None,
        "personalized_execution": None,
        "papo_offline_proxy": None,
        "suggestion_task_readiness": None,
        "execution_reference_retrieval": None,
    }
    if args.suggestion_results:
        report["proactive_suggestion"] = proactive_metrics(read_rows(args.suggestion_results))
    if args.execution_results:
        report["personalized_execution"] = execution_metrics(read_rows(args.execution_results))
    if args.trees:
        report["papo_offline_proxy"] = papo_tree_proxy_metrics(read_rows(args.trees))
    if args.suggestion_tasks:
        report["suggestion_task_readiness"] = suggestion_task_readiness(read_rows(args.suggestion_tasks))
    if args.execution_tasks:
        report["execution_reference_retrieval"] = execution_reference_metrics(read_rows(args.execution_tasks))

    write_json(args.out, report)
    print(f"wrote: {args.out}")
    for name, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
