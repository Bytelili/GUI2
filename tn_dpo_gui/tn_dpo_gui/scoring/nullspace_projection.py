from __future__ import annotations

import numpy as np


def _sanitize(values) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def projection_coefficient(task_margins, preference_margins, weights=None, eps: float = 1e-8) -> float:
    task = _sanitize(task_margins)
    pref = _sanitize(preference_margins)
    if task.shape != pref.shape:
        raise ValueError("task_margins and preference_margins must have the same shape")
    if task.size == 0:
        return 0.0
    weight_array = np.ones_like(task) if weights is None else _sanitize(weights)
    if weight_array.shape != task.shape:
        raise ValueError("weights must match margin shape")
    denominator = float(np.sum(weight_array * task * task) + eps)
    numerator = float(np.sum(weight_array * pref * task))
    return numerator / denominator


def project_preference_to_task_nullspace(task_margins, preference_margins, weights=None, eps: float = 1e-8) -> tuple[float, np.ndarray]:
    task = _sanitize(task_margins)
    pref = _sanitize(preference_margins)
    rho = projection_coefficient(task, pref, weights=weights, eps=eps)
    null_margins = pref - rho * task
    null_margins[np.abs(null_margins) < eps * 10] = 0.0
    return rho, null_margins


def orthogonality_dot(task_margins, null_margins, weights=None) -> float:
    task = _sanitize(task_margins)
    null = _sanitize(null_margins)
    weight_array = np.ones_like(task) if weights is None else _sanitize(weights)
    return float(np.sum(weight_array * task * null))
