from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_policy_logratio(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    return chosen_scores - rejected_scores


def weighted_dpo_loss(
    policy_logratio: torch.Tensor,
    ref_logratio: torch.Tensor,
    beta: float,
    weights: torch.Tensor,
) -> torch.Tensor:
    logits = float(beta) * (policy_logratio - ref_logratio)
    per_example = -F.logsigmoid(logits)
    normalized_weights = weights / weights.clamp_min(1e-8).mean()
    return torch.mean(normalized_weights * per_example)


def capacity_slack(null_margin: float, uncertainty: float, lambda_u: float = 0.5) -> float:
    return max(0.0, abs(float(null_margin)) - float(lambda_u) * float(uncertainty))
