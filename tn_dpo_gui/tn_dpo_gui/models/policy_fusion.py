from __future__ import annotations

import math
from typing import Mapping

import numpy as np


def normalize_probabilities(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    total = float(np.sum(array))
    if total <= 0.0:
        return np.full_like(array, 1.0 / max(len(array), 1), dtype=float)
    return array / total


def fuse_probabilities(base_probs, adapted_probs, gate_value: float):
    base = normalize_probabilities(base_probs)
    adapted = normalize_probabilities(adapted_probs)
    fused = (1.0 - gate_value) * base + gate_value * adapted
    return normalize_probabilities(fused)


def fuse_policy_maps(base_probs: Mapping[str, float], adapted_scores: Mapping[str, float], gate_value: float) -> dict[str, float]:
    keys = list(base_probs.keys())
    adapted_exp = np.asarray([math.exp(adapted_scores.get(key, -10.0)) for key in keys], dtype=float)
    fused = fuse_probabilities(np.asarray([math.exp(base_probs[key]) for key in keys], dtype=float), adapted_exp, gate_value)
    return {key: float(value) for key, value in zip(keys, fused)}
