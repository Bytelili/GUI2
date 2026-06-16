from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize UI-TARS Base/SFT Proactive level metrics.")
    parser.add_argument("--reports-root", default="reports/ui_tars_proactive")
    parser.add_argument("--mode", default="strict_holdout")
    parser.add_argument("--models", nargs="+", default=["ui_tars_7b_base", "ui_tars_7b_sft"])
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    root = Path(args.reports_root)
    output = Path(args.output_dir) if args.output_dir else root / "summary"
    rows = collect_results(root, args.mode, args.models)
    if not rows:
        raise ValueError("No UI-TARS metric reports were found")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{args.mode}_ui_tars_level_results.csv"
    _write_csv(csv_path, rows)
    md_path = output / f"{args.mode}_ui_tars_level_results.md"
    md_path.write_text(_markdown(rows), encoding="utf-8")

    comparison_rows = compare_sft_vs_base(rows)
    comparison_csv = output / f"{args.mode}_ui_tars_sft_minus_base.csv"
    if comparison_rows:
        _write_csv(comparison_csv, comparison_rows)
        comparison_md = output / f"{args.mode}_ui_tars_sft_minus_base.md"
        comparison_md.write_text(_markdown(comparison_rows), encoding="utf-8")

    print(_markdown(rows))
    if comparison_rows:
        print("\n===== SFT - Base =====")
        print(_markdown(comparison_rows))
    print(f"Written: {csv_path}")
    print(f"Written: {md_path}")
    if comparison_rows:
        print(f"Written: {comparison_csv}")


def collect_results(reports_root: Path, mode: str, models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for level in range(4):
            path = reports_root / model / mode / f"level_{level}" / "metrics" / "benchmark_metrics.json"
            if not path.exists():
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            metrics = _select_level_metrics(report.get("proactive_suggestion", {}), level)
            if not metrics:
                continue
            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "level": level,
                    "count": metrics.get("count", ""),
                    "official_similarity": _mean(metrics.get("official_similarity")),
                    "official_similarity_raw": _mean(metrics.get("official_similarity_raw")),
                    "official_similarity_ci95_low": _ci(metrics.get("official_similarity"), "ci95_low"),
                    "official_similarity_ci95_high": _ci(metrics.get("official_similarity"), "ci95_high"),
                    "edit_similarity": _mean(metrics.get("edit_similarity")),
                    "semantic_similarity": _mean(metrics.get("semantic_similarity")),
                    "time_mean": _mean(metrics.get("time")),
                    "time_median": _median(metrics.get("time")),
                    "token_mean": _mean(metrics.get("token")),
                    "token_median": _median(metrics.get("token")),
                    "error_rate": metrics.get("error_rate", ""),
                }
            )
    return rows


def compare_sft_vs_base(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_level_model = {(str(row["model"]), int(row["level"])): row for row in rows}
    result: list[dict[str, Any]] = []
    for level in range(4):
        base = by_level_model.get(("ui_tars_7b_base", level))
        sft = by_level_model.get(("ui_tars_7b_sft", level))
        if not base or not sft:
            continue
        result.append(
            {
                "comparison": "ui_tars_7b_sft_minus_base",
                "mode": sft["mode"],
                "level": level,
                "count": sft["count"],
                "official_similarity_delta": _number(sft["official_similarity"]) - _number(base["official_similarity"]),
                "official_similarity_raw_delta": _number(sft["official_similarity_raw"])
                - _number(base["official_similarity_raw"]),
                "edit_similarity_delta": _number(sft["edit_similarity"]) - _number(base["edit_similarity"]),
                "semantic_similarity_delta": _number(sft["semantic_similarity"])
                - _number(base["semantic_similarity"]),
                "time_mean_delta": _number(sft["time_mean"]) - _number(base["time_mean"]),
                "token_mean_delta": _number(sft["token_mean"]) - _number(base["token_mean"]),
                "error_rate_delta": _number(sft["error_rate"]) - _number(base["error_rate"]),
            }
        )
    return result


def _select_level_metrics(proactive: dict[str, Any], level: int) -> dict[str, Any]:
    if f"level_{level}" in proactive:
        return proactive[f"level_{level}"]
    if len(proactive) == 1:
        return next(iter(proactive.values()))
    return proactive if "official_similarity" in proactive else {}


def _mean(value: Any) -> Any:
    return value.get("mean", "") if isinstance(value, dict) else value


def _median(value: Any) -> Any:
    return value.get("median", "") if isinstance(value, dict) else ""


def _ci(value: Any, key: str) -> Any:
    return value.get(key, "") if isinstance(value, dict) else ""


def _number(value: Any) -> float:
    return float(value) if value != "" else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
