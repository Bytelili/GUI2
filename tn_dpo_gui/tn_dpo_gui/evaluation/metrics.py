from __future__ import annotations

import math

import numpy as np

from tn_dpo_gui.pair_builder.pair_schema import TNDPOPair
from tn_dpo_gui.scoring.nullspace_projection import orthogonality_dot


def pair_accuracy_from_scores(chosen_scores, rejected_scores) -> float:
    chosen = np.asarray(chosen_scores, dtype=float)
    rejected = np.asarray(rejected_scores, dtype=float)
    if chosen.size == 0:
        return 0.0
    return float(np.mean(chosen > rejected))


def weighted_pair_accuracy(pairs: list[TNDPOPair]) -> float:
    if not pairs:
        return 0.0
    weights = np.asarray([pair.weight for pair in pairs], dtype=float)
    correct = np.asarray([1.0 if pair.null_margin > 0 else 0.0 for pair in pairs], dtype=float)
    if float(weights.sum()) <= 0.0:
        return float(correct.mean())
    return float(np.sum(weights * correct) / np.sum(weights))


def weighted_accuracy_from_scores(chosen_scores, rejected_scores, weights) -> float:
    chosen = np.asarray(chosen_scores, dtype=float)
    rejected = np.asarray(rejected_scores, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if chosen.size == 0:
        return 0.0
    correct = (chosen > rejected).astype(float)
    if float(weight_array.sum()) <= 0.0:
        return float(np.mean(correct))
    return float(np.sum(weight_array * correct) / np.sum(weight_array))


def safety_metric(pairs: list[TNDPOPair]) -> float:
    if not pairs:
        return 0.0
    safe = [1.0 if pair.task_margin >= -0.05 else 0.0 for pair in pairs]
    return float(sum(safe) / len(safe))


def preference_proxy(pairs: list[TNDPOPair]) -> float:
    if not pairs:
        return 0.0
    return float(sum(pair.preference_margin for pair in pairs) / len(pairs))


def projection_metrics(pairs: list[TNDPOPair]) -> dict[str, float]:
    if not pairs:
        return {"rho_mean": 0.0, "orthogonality_dot": 0.0, "corr_task_null": 0.0}
    task = np.asarray([pair.task_margin for pair in pairs], dtype=float)
    null = np.asarray([pair.null_margin for pair in pairs], dtype=float)
    weights = np.asarray([pair.weight for pair in pairs], dtype=float)
    corr = 0.0
    if task.size > 1 and np.std(task) > 0 and np.std(null) > 0:
        corr = float(np.corrcoef(task, null)[0, 1])
    return {
        "rho_mean": float(np.mean([pair.projection_rho for pair in pairs])),
        "orthogonality_dot": orthogonality_dot(task, null, weights=weights),
        "corr_task_null": corr,
    }


def regression_metrics(predictions, targets) -> dict[str, float]:
    preds = np.asarray(predictions, dtype=float)
    gold = np.asarray(targets, dtype=float)
    if preds.size == 0:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0}
    errors = preds - gold
    mse = float(np.mean(errors * errors))
    return {"mae": float(np.mean(np.abs(errors))), "mse": mse, "rmse": math.sqrt(mse)}
