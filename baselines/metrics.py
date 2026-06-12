from __future__ import annotations

from collections import defaultdict
from math import dist
from statistics import mean
from typing import Any

from .common import action_type, levenshtein_similarity, parse_action, sequence_similarity


def retrieval_report(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for task in tasks:
        target = list(task.get("target_actions") or [])
        for mode, refs in dict(task.get("references") or {}).items():
            if not refs:
                continue
            reference = list(refs[0].get("actions") or [])
            grouped[mode].append(
                {
                    "sequence_similarity": sequence_similarity(target, reference),
                    "action_levenshtein_similarity": levenshtein_similarity(target, reference),
                    "intent_similarity": float(refs[0].get("intent_similarity", 0.0) or 0.0),
                }
            )
    modes = {}
    for mode, rows in grouped.items():
        metrics = {
            key: mean(row[key] for row in rows)
            for key in rows[0]
        } if rows else {}
        modes[mode] = {
            "num_episodes": len(rows),
            "coverage": len(rows) / max(len(tasks), 1),
            **metrics,
        }
    same = modes.get("same_user_top1", {})
    cross = modes.get("cross_user_top1", {})
    if same and cross:
        modes["same_vs_cross"] = {
            "sequence_similarity_gain": same["sequence_similarity"] - cross["sequence_similarity"],
            "levenshtein_similarity_gain": (
                same["action_levenshtein_similarity"] - cross["action_levenshtein_similarity"]
            ),
        }
    return {"num_tasks": len(tasks), "modes": modes}


def prediction_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("prediction") is not None and row.get("target_action")]
    if not valid_rows:
        return {"num_rows": len(rows), "num_evaluated": 0}
    metrics: list[dict[str, float]] = []
    sequences: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: {"prediction": [], "target": [], "cross": []}
    )
    for row in valid_rows:
        prediction = str(row["prediction"])
        target = str(row["target_action"])
        parsed_prediction = parse_action(prediction)
        parsed_target = parse_action(target)
        coordinate_distance = None
        if parsed_prediction.get("coordinates") and parsed_target.get("coordinates"):
            coordinate_distance = dist(parsed_prediction["coordinates"], parsed_target["coordinates"])
        metrics.append(
            {
                "parse_valid": float(bool(parsed_prediction["valid"])),
                "type_accuracy": float(action_type(prediction) == action_type(target)),
                "exact_accuracy": float(prediction.strip() == target.strip()),
                "coordinate_distance": float(coordinate_distance) if coordinate_distance is not None else 0.0,
                "has_coordinate_distance": float(coordinate_distance is not None),
            }
        )
        key = (str(row.get("variant") or ""), str(row.get("episode_id") or ""))
        sequences[key]["prediction"].append(prediction)
        sequences[key]["target"].append(target)
        sequences[key]["cross"] = list(row.get("cross_user_actions") or [])

    sequence_rows = []
    for (variant, episode_id), item in sequences.items():
        up_sim = sequence_similarity(item["prediction"], item["target"])
        down_sim = sequence_similarity(item["prediction"], item["cross"]) if item["cross"] else 0.0
        sim2_denominator = down_sim if down_sim > 0 else 0.4
        sequence_rows.append(
            {
                "variant": variant,
                "episode_id": episode_id,
                "up_sim": up_sim,
                "down_sim": down_sim,
                "sim2": up_sim / sim2_denominator,
            }
        )
    coord_rows = [row["coordinate_distance"] for row in metrics if row["has_coordinate_distance"]]
    return {
        "num_rows": len(rows),
        "num_evaluated": len(valid_rows),
        "parse_valid_rate": mean(row["parse_valid"] for row in metrics),
        "action_type_accuracy": mean(row["type_accuracy"] for row in metrics),
        "exact_action_accuracy": mean(row["exact_accuracy"] for row in metrics),
        "mean_coordinate_distance": mean(coord_rows) if coord_rows else None,
        "sequence_up_sim": mean(row["up_sim"] for row in sequence_rows) if sequence_rows else None,
        "sequence_down_sim": mean(row["down_sim"] for row in sequence_rows) if sequence_rows else None,
        "sequence_sim2": mean(row["sim2"] for row in sequence_rows) if sequence_rows else None,
    }
