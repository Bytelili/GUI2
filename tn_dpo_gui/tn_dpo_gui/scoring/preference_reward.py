from __future__ import annotations

import numpy as np

from tn_dpo_gui.utils.math_utils import cosine_similarity


def preference_score(user_vector: np.ndarray, trajectory_vector: np.ndarray, general_vector: np.ndarray | None = None, general_weight: float = 0.0) -> float:
    score = cosine_similarity(user_vector, trajectory_vector)
    if general_vector is not None and general_weight:
        score -= general_weight * cosine_similarity(general_vector, trajectory_vector)
    return float(score)


def preference_margin(left_score: float, right_score: float) -> float:
    return float(left_score) - float(right_score)
