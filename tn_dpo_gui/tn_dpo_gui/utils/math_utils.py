from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def softmax(values: Iterable[float], temperature: float = 1.0) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return array
    temperature = max(float(temperature), 1e-8)
    shifted = (array - np.max(array)) / temperature
    exp_values = np.exp(shifted)
    return exp_values / np.clip(exp_values.sum(), 1e-8, None)


def safe_divide(numerator: float, denominator: float, eps: float = 1e-8) -> float:
    return float(numerator) / float(denominator if abs(denominator) > eps else math.copysign(eps, denominator or 1.0))


def l2_normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        return np.zeros_like(vector)
    return vector / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray, eps: float = 1e-8) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= eps or right_norm <= eps:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))
