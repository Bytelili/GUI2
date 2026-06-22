from __future__ import annotations

from tn_dpo_gui.data.schema import TrajectoryContinuation


def task_reward(
    task_success: float,
    progress: float,
    invalid_count: int,
    risk_score: float,
    eta: float = 1.0,
    mu: float = 0.25,
    risk_coef: float = 0.5,
) -> float:
    return float(task_success) + float(eta) * float(progress) - float(mu) * float(invalid_count) - float(risk_coef) * float(risk_score)


def continuation_task_reward(continuation: TrajectoryContinuation, eta: float = 1.0, mu: float = 0.25, risk_coef: float = 0.5) -> float:
    return task_reward(
        continuation.task_success,
        continuation.progress,
        continuation.invalid_count,
        continuation.risk_score,
        eta=eta,
        mu=mu,
        risk_coef=risk_coef,
    )


def continuation_task_distance(
    left: TrajectoryContinuation,
    right: TrajectoryContinuation,
    success_weight: float = 1.0,
    progress_weight: float = 1.0,
    goal_weight: float = 0.5,
    invalid_weight: float = 0.25,
    risk_weight: float = 0.5,
) -> float:
    goal_mismatch = 0.0 if (left.goal_state or "") == (right.goal_state or "") else 1.0
    return (
        success_weight * abs(left.task_success - right.task_success)
        + progress_weight * abs(left.progress - right.progress)
        + goal_weight * goal_mismatch
        + invalid_weight * abs(left.invalid_count - right.invalid_count)
        + risk_weight * abs(left.risk_score - right.risk_score)
    )
