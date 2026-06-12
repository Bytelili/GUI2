from __future__ import annotations

from collections import Counter
from typing import Any

from .io import action_label


DEFAULT_WEIGHTS = {
    "target": 0.35,
    "order": 0.25,
    "habit": 0.25,
    "avoid": 0.15,
}


def build_user_profile(root_step: dict[str, Any], all_steps: list[dict[str, Any]]) -> dict[str, Any]:
    user_id = str(root_step.get("user_id") or "")
    root_rank = int(root_step.get("chronological_rank", 0) or 0)
    past = [
        row for row in all_steps
        if str(row.get("user_id") or "") == user_id
        and int(row.get("chronological_rank", 0) or 0) < root_rank
    ]
    actions = [action_label(row) for row in past if action_label(row)]
    action_counts = Counter(actions)
    bigrams = Counter(zip(actions, actions[1:]))
    return {
        "num_history_steps": len(actions),
        "action_counts": dict(action_counts),
        "bigram_counts": {"\t".join(k): v for k, v in bigrams.items()},
        "max_action_count": max(action_counts.values(), default=0),
    }


def lcs_ratio(left: list[str], right: list[str]) -> float:
    if not left:
        return 0.0
    if not right:
        return 0.0
    prev = [0] * (len(right) + 1)
    for a in left:
        cur = [0] * (len(right) + 1)
        for j, b in enumerate(right, 1):
            cur[j] = prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1] / max(len(left), 1)


def score_leaf(
    actions: list[str],
    target_actions: list[str],
    path: list[dict[str, Any]],
    user_profile: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    weights = weights or DEFAULT_WEIGHTS
    prefix_target = target_actions[: len(actions)]
    exact_matches = sum(1 for a, b in zip(actions, prefix_target) if a == b)
    target_score = exact_matches / max(len(actions), 1)
    order_score = lcs_ratio(actions, prefix_target)

    counts = user_profile.get("action_counts") if isinstance(user_profile.get("action_counts"), dict) else {}
    max_count = float(user_profile.get("max_action_count", 0) or 0)
    if max_count > 0 and actions:
        habit_score = sum(float(counts.get(a, 0) or 0) / max_count for a in actions) / len(actions)
    else:
        # Cold-start neutral value: do not punish the first few episodes for lacking history.
        habit_score = 0.5

    cross_edges = sum(1 for edge in path if "cross_user" in str(edge.get("source", "")))
    avoid_score = 1.0 - cross_edges / max(len(path), 1)

    total = (
        weights.get("target", 0.0) * target_score
        + weights.get("order", 0.0) * order_score
        + weights.get("habit", 0.0) * habit_score
        + weights.get("avoid", 0.0) * avoid_score
    )
    return {
        "target": target_score,
        "order": order_score,
        "habit": habit_score,
        "avoid": avoid_score,
        "total": total,
        "weights": dict(weights),
    }

