from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


LEVELS = (0, 1, 2, 3)
COMPARISONS = ((0, 1), (1, 2), (2, 3), (0, 3))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paper-ready Proactive screenshot-level report.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--scored-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = build_report(
        args.metrics,
        args.scored_predictions,
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("PROACTIVE LEVEL REPORT PASSED")


def build_report(
    metrics_path: str | Path,
    scored_path: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    metrics_source = Path(metrics_path).resolve()
    scored_source = Path(scored_path).resolve()
    metrics = json.loads(metrics_source.read_text(encoding="utf-8"))["proactive_suggestion"]
    rows = read_csv(scored_source)
    by_level: dict[int, dict[str, dict[str, str]]] = {}
    for level in LEVELS:
        selected = [row for row in rows if int(float(row["level"])) == level]
        identifiers = [str(row.get("task_id") or "") for row in selected]
        if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Level-{level} contains empty or duplicate task IDs")
        if any(str(row.get("error") or "").strip() for row in selected):
            raise ValueError(f"Level-{level} contains failed predictions")
        by_level[level] = {identifier: row for identifier, row in zip(identifiers, selected)}
    common_ids = set(by_level[0])
    if not common_ids or any(set(by_level[level]) != common_ids for level in LEVELS[1:]):
        raise ValueError("All screenshot levels must contain the same non-empty task-ID set")

    summaries = [level_summary(level, metrics[f"level_{level}"]) for level in LEVELS]
    comparisons = [
        paired_comparison(
            by_level[lower],
            by_level[upper],
            lower=lower,
            upper=upper,
            samples=bootstrap_samples,
            seed=seed,
        )
        for lower, upper in COMPARISONS
    ]
    report = {
        "status": "passed",
        "paired_tasks": len(common_ids),
        "inputs": {
            "metrics_path": str(metrics_source),
            "metrics_sha256": sha256_file(metrics_source),
            "scored_predictions_path": str(scored_source),
            "scored_predictions_sha256": sha256_file(scored_source),
        },
        "levels": summaries,
        "paired_comparisons": comparisons,
        "interpretation": {
            "primary_metric": "official_similarity",
            "main_protocol": "strict_holdout",
            "statistical_rule": "A paired gain is supported when its 95% CI is entirely above zero.",
        },
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "proactive_level_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output / "level_summary.csv", summaries)
    write_csv(output / "paired_level_comparisons.csv", comparisons)
    (output / "proactive_level_report.md").write_text(markdown_report(report), encoding="utf-8")
    return report


def level_summary(level: int, metrics: dict[str, Any]) -> dict[str, Any]:
    official = metrics["official_similarity"]
    return {
        "level": level,
        "screenshots": level,
        "samples": int(metrics["count"]),
        "official_similarity_mean": float(official["mean"]),
        "official_similarity_ci95_low": float(official["ci95_low"]),
        "official_similarity_ci95_high": float(official["ci95_high"]),
        "edit_similarity_mean": float(metrics["edit_similarity"]["mean"]),
        "semantic_similarity_mean": float(metrics["semantic_similarity"]["mean"]),
        "time_mean": float(metrics["time"]["mean"]),
        "token_mean": float(metrics["token"]["mean"]),
        "error_rate": float(metrics["error_rate"]),
    }


def paired_comparison(
    lower_rows: dict[str, dict[str, str]],
    upper_rows: dict[str, dict[str, str]],
    *,
    lower: int,
    upper: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    identifiers = sorted(lower_rows)
    deltas = [
        number(upper_rows[identifier]["official_similarity"])
        - number(lower_rows[identifier]["official_similarity"])
        for identifier in identifiers
    ]
    time_deltas = [
        number(upper_rows[identifier]["time"]) - number(lower_rows[identifier]["time"])
        for identifier in identifiers
    ]
    token_deltas = [
        number(upper_rows[identifier]["token"]) - number(lower_rows[identifier]["token"])
        for identifier in identifiers
    ]
    task_low, task_high = bootstrap_ci(deltas, samples, f"{seed}:task:{lower}:{upper}")
    by_user: dict[str, list[float]] = defaultdict(list)
    for identifier, delta in zip(identifiers, deltas):
        by_user[str(upper_rows[identifier].get("user_id") or "")].append(delta)
    user_means = [statistics.fmean(values) for values in by_user.values()]
    user_low, user_high = bootstrap_ci(user_means, samples, f"{seed}:user:{lower}:{upper}")
    mean_delta = statistics.fmean(deltas)
    mean_tokens = statistics.fmean(token_deltas)
    return {
        "comparison": f"level_{upper}_vs_level_{lower}",
        "reference_level": lower,
        "candidate_level": upper,
        "paired_tasks": len(identifiers),
        "user_clusters": len(user_means),
        "mean_similarity_delta": mean_delta,
        "task_bootstrap_ci95_low": task_low,
        "task_bootstrap_ci95_high": task_high,
        "macro_user_mean_delta": statistics.fmean(user_means),
        "user_cluster_ci95_low": user_low,
        "user_cluster_ci95_high": user_high,
        "win_rate": sum(value > 0 for value in deltas) / len(deltas),
        "tie_rate": sum(value == 0 for value in deltas) / len(deltas),
        "loss_rate": sum(value < 0 for value in deltas) / len(deltas),
        "mean_time_delta": statistics.fmean(time_deltas),
        "mean_token_delta": mean_tokens,
        "similarity_gain_per_1000_added_tokens": (
            mean_delta / mean_tokens * 1000.0 if mean_tokens > 0 else None
        ),
    }


def bootstrap_ci(values: list[float], samples: int, seed: str) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(samples)
    )
    return percentile(draws, 0.025), percentile(draws, 0.975)


def percentile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Proactive SFT Strict-Holdout Screenshot-Level Report",
        "",
        f"All comparisons use the same {report['paired_tasks']} complete test episodes.",
        "",
        "## Official Metrics",
        "",
        "| Level | N | Official Similarity | 95% CI | Edit Sim. | Semantic Sim. | Time | Tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["levels"]:
        lines.append(
            f"| {row['level']} | {row['samples']} | {fmt(row['official_similarity_mean'])} | "
            f"[{fmt(row['official_similarity_ci95_low'])}, {fmt(row['official_similarity_ci95_high'])}] | "
            f"{fmt(row['edit_similarity_mean'])} | {fmt(row['semantic_similarity_mean'])} | "
            f"{fmt(row['time_mean'])} | {fmt(row['token_mean'], 1)} |"
        )
    lines.extend(
        [
            "",
            "## Paired Screenshot Gains",
            "",
            "| Comparison | Paired N | Mean Gain | Task 95% CI | User-Cluster 95% CI | Win Rate | Time Delta | Token Delta | Gain / 1K Tokens |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["paired_comparisons"]:
        lines.append(
            f"| {row['comparison']} | {row['paired_tasks']} | {fmt(row['mean_similarity_delta'])} | "
            f"[{fmt(row['task_bootstrap_ci95_low'])}, {fmt(row['task_bootstrap_ci95_high'])}] | "
            f"[{fmt(row['user_cluster_ci95_low'])}, {fmt(row['user_cluster_ci95_high'])}] | "
            f"{fmt(row['win_rate'])} | {fmt(row['mean_time_delta'])} | "
            f"{fmt(row['mean_token_delta'], 1)} | {fmt(row['similarity_gain_per_1000_added_tokens'])} |"
        )
    lines.extend(
        [
            "",
            "## Reporting Rule",
            "",
            "Use Strict Holdout as the main result. Treat Official Online as a separate official-protocol reproduction.",
            "",
        ]
    )
    return "\n".join(lines)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {value!r}")
    return result


def fmt(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
