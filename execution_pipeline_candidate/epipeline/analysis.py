from __future__ import annotations

import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import read_csv, write_csv, write_json


METRICS = {
    "success": "higher",
    "up_sim": "higher",
    "down_sim": "lower",
    "similarity": "higher",
    "step_ratio": "lower",
    "time": "lower",
    "token": "lower",
    "strict_action_up_sim": "higher",
    "strict_action_down_sim": "lower",
    "strict_action_similarity": "higher",
}


def analyze_manifest(manifest: dict[str, Any], *, bootstrap_samples: int = 10000) -> dict[str, Any]:
    run_rows: dict[str, list[dict[str, str]]] = {}
    for entry in manifest["runs"]:
        path = Path(str(entry["run_dir"])) / "execution_results_scored.csv"
        if path.exists():
            run_rows[str(entry["id"])] = read_csv(path)
    comparisons: list[dict[str, Any]] = []
    user_effects: list[dict[str, Any]] = []
    for spec in manifest.get("comparisons") or []:
        reference_id = str(spec["reference_run"])
        candidate_id = str(spec["candidate_run"])
        if reference_id not in run_rows or candidate_id not in run_rows:
            continue
        for metric, direction in METRICS.items():
            comparison, per_user = paired_comparison(
                run_rows[reference_id],
                run_rows[candidate_id],
                metric=metric,
                direction=direction,
                samples=bootstrap_samples,
                seed=int(manifest["seed"]),
            )
            comparisons.append({"comparison_id": spec["id"], **comparison})
            user_effects.extend(
                {
                    "comparison_id": spec["id"],
                    "metric": metric,
                    "direction": direction,
                    **row,
                }
                for row in per_user
            )
    report = {
        "status": "completed",
        "bootstrap_samples": bootstrap_samples,
        "seed": int(manifest["seed"]),
        "available_runs": sorted(run_rows),
        "paired_comparisons": comparisons,
        "user_effects": user_effects,
    }
    output = Path(str(manifest["output_root"]))
    write_json(output / "analysis_report.json", report)
    write_csv(output / "paired_comparisons.csv", comparisons)
    write_csv(output / "user_effects.csv", user_effects)
    return report


def paired_comparison(
    reference_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    *,
    metric: str,
    direction: str,
    samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference = eligible_values(reference_rows, metric)
    candidate = eligible_values(candidate_rows, metric)
    common = sorted(set(reference) & set(candidate))
    signed = [
        (candidate[identifier][0] - reference[identifier][0]) * (1 if direction == "higher" else -1)
        for identifier in common
    ]
    result = {
        "reference_run": reference_rows[0].get("run_id", "") if reference_rows else "",
        "candidate_run": candidate_rows[0].get("run_id", "") if candidate_rows else "",
        "metric": metric,
        "direction": direction,
        "reference_tasks": len(reference),
        "candidate_tasks": len(candidate),
        "paired_tasks": len(common),
        "coverage_match": set(reference) == set(candidate),
        "mean_improvement": average(signed),
        "ci95_low": None,
        "ci95_high": None,
        "win_rate": sum(value > 0 for value in signed) / len(signed) if signed else None,
    }
    if signed:
        rng = random.Random(f"{seed}:{metric}:{result['reference_run']}:{result['candidate_run']}")
        draws = sorted(
            statistics.fmean(rng.choice(signed) for _ in signed)
            for _ in range(samples)
        )
        result["ci95_low"] = percentile(draws, 0.025)
        result["ci95_high"] = percentile(draws, 0.975)
    by_user: dict[str, list[float]] = defaultdict(list)
    for identifier, value in zip(common, signed):
        user = candidate[identifier][1] or reference[identifier][1]
        by_user[user].append(value)
    per_user = [
        {
            "user_id": user,
            "paired_tasks": len(values),
            "mean_improvement": statistics.fmean(values),
            "win_rate": sum(value > 0 for value in values) / len(values),
        }
        for user, values in sorted(by_user.items())
    ]
    result["macro_user_mean_improvement"] = average([row["mean_improvement"] for row in per_user])
    result["worst_user_improvement"] = min((row["mean_improvement"] for row in per_user), default=None)
    result["fraction_users_improved"] = (
        sum(row["mean_improvement"] > 0 for row in per_user) / len(per_user) if per_user else None
    )
    result["user_clusters"] = len(per_user)
    result["macro_user_ci95_low"] = None
    result["macro_user_ci95_high"] = None
    if per_user:
        user_means = [float(row["mean_improvement"]) for row in per_user]
        rng = random.Random(f"{seed}:user-cluster:{metric}:{result['reference_run']}:{result['candidate_run']}")
        draws = sorted(
            statistics.fmean(rng.choice(user_means) for _ in user_means)
            for _ in range(samples)
        )
        result["macro_user_ci95_low"] = percentile(draws, 0.025)
        result["macro_user_ci95_high"] = percentile(draws, 0.975)
    return result, per_user


def eligible_values(rows: list[dict[str, str]], metric: str) -> dict[str, tuple[float, str]]:
    output: dict[str, tuple[float, str]] = {}
    for row in rows:
        if metric == "success" and str(row.get("success_verified") or "").lower() not in {"true", "1", "yes"}:
            continue
        identifier = str(row.get("task_id") or "")
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if identifier:
            output[identifier] = (value, str(row.get("user_id") or ""))
    return output


def average(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def percentile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction
