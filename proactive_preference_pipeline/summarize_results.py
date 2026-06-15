from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize official Level 0-3 metrics for preference models.")
    parser.add_argument("--reports-root", default="reports/proactive_preference")
    parser.add_argument("--mode", default="strict_holdout")
    parser.add_argument("--models", nargs="+", default=["listwise", "dpo"])
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    reports_root = Path(args.reports_root)
    output = Path(args.output_dir) if args.output_dir else reports_root / "summary"
    rows = collect_results(reports_root, args.mode, args.models)
    if not rows:
        raise ValueError("No completed preference-model metric reports were found")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{args.mode}_level_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = _markdown(rows)
    md_path = output / f"{args.mode}_level_results.md"
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Written: {csv_path}")
    print(f"Written: {md_path}")


def collect_results(reports_root: Path, mode: str, models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for level in range(4):
            path = reports_root / model / mode / f"level_{level}" / "metrics" / "benchmark_metrics.json"
            if not path.exists():
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            proactive = report.get("proactive_suggestion", {})
            metrics = _select_level_metrics(proactive, level)
            if not metrics:
                continue
            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "level": level,
                    "count": metrics.get("count", ""),
                    "official_similarity": _mean(metrics.get("official_similarity")),
                    "edit_similarity": _mean(metrics.get("edit_similarity")),
                    "semantic_similarity": _mean(metrics.get("semantic_similarity")),
                    "time": _mean(metrics.get("time")),
                    "token": _mean(metrics.get("token")),
                    "error_rate": metrics.get("error_rate", ""),
                }
            )
    return rows


def _select_level_metrics(proactive: dict[str, Any], level: int) -> dict[str, Any]:
    if f"level_{level}" in proactive:
        return proactive[f"level_{level}"]
    if len(proactive) == 1:
        return next(iter(proactive.values()))
    return proactive if "official_similarity" in proactive else {}


def _mean(value: Any) -> Any:
    return value.get("mean", "") if isinstance(value, dict) else value


def _markdown(rows: list[dict[str, Any]]) -> str:
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row[key]) for key in headers) + " |")
    return "\n".join(lines) + "\n"


def _format(value: Any) -> str:
    return f"{value:.6f}" if isinstance(value, float) else str(value)


if __name__ == "__main__":
    main()
