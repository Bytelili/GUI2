from __future__ import annotations

import numpy as np

from tn_dpo_gui.data.schema import TrajectoryContinuation
from tn_dpo_gui.utils.math_utils import sigmoid

from .task_reward import continuation_task_reward


def continuation_uncertainty(continuations: list[TrajectoryContinuation]) -> float:
    if not continuations:
        return 1.0
    rewards = np.asarray([continuation_task_reward(item) for item in continuations], dtype=float)
    retrieval_scores = np.asarray([item.retrieval_score for item in continuations], dtype=float)
    reward_std = float(np.std(rewards)) if len(rewards) > 1 else 0.0
    retrieval_penalty = max(0.0, 1.0 - float(np.max(retrieval_scores)))
    branch_penalty = 0.05 * max(0, len(continuations) - 1)
    return reward_std + retrieval_penalty + branch_penalty


def pair_uncertainty(left_candidates: list[TrajectoryContinuation], right_candidates: list[TrajectoryContinuation]) -> float:
    return 0.5 * (continuation_uncertainty(left_candidates) + continuation_uncertainty(right_candidates))


def pair_weight(null_margin: float, uncertainty: float, lambda_u: float = 0.5, tau_omega: float = 0.25) -> float:
    score = (abs(float(null_margin)) - float(lambda_u) * float(uncertainty)) / max(float(tau_omega), 1e-8)
    return float(sigmoid(score))
