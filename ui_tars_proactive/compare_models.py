from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


LEVELS = (0, 1, 2, 3)
SCORED_NAME = "metrics/proactive_predictions_scored.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired comparison of two Proactive UI-TARS adapters.")
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-limit", type=int, default=100)
    args = parser.parse_args()

    report = compare_models(
        Path(args.reference_root),
        Path(args.candidate_root),
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
        output_dir=Path(args.output_dir),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        case_limit=args.case_limit,
    )
    print(markdown_report(report))
    print("PAIRED MODEL COMPARISON PASSED")


def compare_models(
    reference_root: Path,
    candidate_root: Path,
    *,
    reference_label: str,
    candidate_label: str,
    output_dir: Path,
    bootstrap_samples: int,
    seed: int,
    case_limit: int,
) -> dict[str, Any]:
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    paired_by_level: dict[int, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []

    for level in LEVELS:
        reference = indexed_rows(reference_root / f"level_{level}" / SCORED_NAME)
        candidate = indexed_rows(candidate_root / f"level_{level}" / SCORED_NAME)
        if set(reference) != set(candidate):
            raise ValueError(
                f"Level-{level} task sets differ: reference={len(reference)}, candidate={len(candidate)}, "
                f"reference_only={len(set(reference) - set(candidate))}, "
                f"candidate_only={len(set(candidate) - set(reference))}"
            )
        pairs = [paired_row(level, reference[task_id], candidate[task_id]) for task_id in sorted(reference)]
        paired_by_level[level] = pairs
        summaries.append(summarize_pairs(pairs, level, bootstrap_samples, f"{seed}:level:{level}"))

    task_ids = set(row["task_id"] for row in paired_by_level[0])
    if any(set(row["task_id"] for row in paired_by_level[level]) != task_ids for level in LEVELS[1:]):
        raise ValueError("All four levels must contain the same task IDs")
    by_task = defaultdict(list)
    for pairs in paired_by_level.values():
        for row in pairs:
            by_task[row["task_id"]].append(row)
    macro_pairs = [macro_task_row(rows) for _, rows in sorted(by_task.items())]
    macro = summarize_pairs(macro_pairs, "macro_0_3", bootstrap_samples, f"{seed}:macro")

    all_pairs = [row for level in LEVELS for row in paired_by_level[level]]
    regressions = sorted(all_pairs, key=lambda row: (row["official_delta"], row["task_id"]))[:case_limit]
    improvements = sorted(all_pairs, key=lambda row: (-row["official_delta"], row["task_id"]))[:case_limit]
    report = {
        "status": "passed",
        "reference_model": reference_label,
        "candidate_model": candidate_label,
        "metric_direction": "candidate_minus_reference",
        "levels": summaries,
        "macro_across_levels": macro,
        "interpretation": {
            "task_significance": "supported when task_bootstrap_ci95 excludes zero",
            "user_significance": "supported when user_cluster_ci95 excludes zero",
            "regression": "negative official_delta means the candidate is worse",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired_model_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "paired_model_comparison.csv", summaries + [macro])
    write_csv(output_dir / "regression_cases.csv", regressions)
    write_csv(output_dir / "improvement_cases.csv", improvements)
    (output_dir / "paired_model_comparison.md").write_text(markdown_report(report), encoding="utf-8")
    return report


def indexed_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing scored predictions: {path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = [dict(row) for row in csv.DictReader(file)]
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in result:
            raise ValueError(f"Empty or duplicate task_id in {path}: {task_id!r}")
        if str(row.get("predicted_intent") or "").upper() == "ERROR" or str(row.get("error") or "").strip():
            raise ValueError(f"Failed prediction in {path}: {task_id}")
        result[task_id] = row
    if not result:
        raise ValueError(f"No scored predictions: {path}")
    return result


def paired_row(level: int, reference: dict[str, str], candidate: dict[str, str]) -> dict[str, Any]:
    if str(reference.get("original_intent") or "") != str(candidate.get("original_intent") or ""):
        raise ValueError(f"Ground truth differs for task: {reference.get('task_id')}")
    return {
        "level": level,
        "task_id": reference["task_id"],
        "user_id": str(reference.get("user_id") or candidate.get("user_id") or ""),
        "original_intent": reference.get("original_intent", ""),
        "reference_prediction": reference.get("predicted_intent", ""),
        "candidate_prediction": candidate.get("predicted_intent", ""),
        "reference_official": number(reference["official_similarity"]),
        "candidate_official": number(candidate["official_similarity"]),
        "official_delta": number(candidate["official_similarity"]) - number(reference["official_similarity"]),
        "edit_delta": number(candidate["edit_similarity"]) - number(reference["edit_similarity"]),
        "semantic_delta": number(candidate["semantic_similarity"]) - number(reference["semantic_similarity"]),
        "time_delta": number(candidate["time"]) - number(reference["time"]),
        "token_delta": number(candidate["token"]) - number(reference["token"]),
    }


def macro_task_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(LEVELS):
        raise ValueError(f"Task does not have all four levels: {rows[0]['task_id']}")
    return {
        "task_id": rows[0]["task_id"],
        "user_id": rows[0]["user_id"],
        **{
            key: statistics.fmean(float(row[key]) for row in rows)
            for key in ["official_delta", "edit_delta", "semantic_delta", "time_delta", "token_delta"]
        },
    }


def summarize_pairs(
    pairs: list[dict[str, Any]], level: int | str, samples: int, seed: str
) -> dict[str, Any]:
    deltas = [float(row["official_delta"]) for row in pairs]
    task_low, task_high = bootstrap_ci(deltas, samples, f"{seed}:task")
    by_user: dict[str, list[float]] = defaultdict(list)
    for row in pairs:
        by_user[str(row["user_id"])].append(float(row["official_delta"]))
    user_means = [statistics.fmean(values) for values in by_user.values()]
    user_low, user_high = bootstrap_ci(user_means, samples, f"{seed}:user")
    return {
        "level": level,
        "paired_tasks": len(pairs),
        "user_clusters": len(user_means),
        "official_delta": statistics.fmean(deltas),
        "task_bootstrap_ci95_low": task_low,
        "task_bootstrap_ci95_high": task_high,
        "macro_user_delta": statistics.fmean(user_means),
        "user_cluster_ci95_low": user_low,
        "user_cluster_ci95_high": user_high,
        "edit_delta": statistics.fmean(float(row["edit_delta"]) for row in pairs),
        "semantic_delta": statistics.fmean(float(row["semantic_delta"]) for row in pairs),
        "win_rate": sum(value > 0 for value in deltas) / len(deltas),
        "tie_rate": sum(value == 0 for value in deltas) / len(deltas),
        "loss_rate": sum(value < 0 for value in deltas) / len(deltas),
        "time_delta": statistics.fmean(float(row["time_delta"]) for row in pairs),
        "token_delta": statistics.fmean(float(row["token_delta"]) for row in pairs),
    }


def bootstrap_ci(values: list[float], samples: int, seed: str) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples))
    return percentile(draws, 0.025), percentile(draws, 0.975)


def percentile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['candidate_model']} vs {report['reference_model']}",
        "",
        "All deltas are candidate minus reference.",
        "",
        "| Level | N | Official Delta | Task 95% CI | User 95% CI | Edit Delta | Semantic Delta | Win | Tie | Loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["levels"] + [report["macro_across_levels"]]:
        lines.append(
            f"| {row['level']} | {row['paired_tasks']} | {fmt(row['official_delta'])} | "
            f"[{fmt(row['task_bootstrap_ci95_low'])}, {fmt(row['task_bootstrap_ci95_high'])}] | "
            f"[{fmt(row['user_cluster_ci95_low'])}, {fmt(row['user_cluster_ci95_high'])}] | "
            f"{fmt(row['edit_delta'])} | {fmt(row['semantic_delta'])} | "
            f"{fmt(row['win_rate'])} | {fmt(row['tie_rate'])} | {fmt(row['loss_rate'])} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value: {value!r}")
    return result


def fmt(value: Any) -> str:
    return f"{float(value):.6f}"


if __name__ == "__main__":
    main()
