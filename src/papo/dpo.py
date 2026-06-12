from __future__ import annotations

import math
from itertools import combinations
from typing import Any


def build_pairs(
    advantage_rows: list[dict[str, Any]],
    margin: float = 0.05,
    tau_m: float = 0.2,
    w_max: float = 5.0,
    beta: float = 0.1,
) -> list[dict[str, Any]]:
    by_node: dict[str, list[dict[str, Any]]] = {}
    for row in advantage_rows:
        key = str(row.get("tree_id") or "") + "::" + str(row.get("node_id") or "")
        by_node.setdefault(key, []).append(row)

    pairs: list[dict[str, Any]] = []
    for node_id, rows in by_node.items():
        for left, right in combinations(rows, 2):
            gap = float(left.get("a_delta", 0.0)) - float(right.get("a_delta", 0.0))
            if abs(gap) <= margin:
                continue
            pos, neg = (left, right) if gap > 0 else (right, left)
            m = abs(gap)
            pairs.append(
                {
                    "node_id": node_id,
                    "step_id": pos.get("step_id", ""),
                    "user_id": pos.get("user_id", ""),
                    "episode_id": pos.get("episode_id", ""),
                    "prefix_actions": pos.get("prefix_actions", []),
                    "state_key": pos.get("state_key", ""),
                    "positive_action": pos.get("action", ""),
                    "negative_action": neg.get("action", ""),
                    "advantage_gap": m,
                    "target_preference_probability": _sigmoid(m / max(beta, 1e-8)),
                    "weight": min(w_max, max(0.0, m / max(tau_m, 1e-8))),
                    "positive_source": pos.get("source", ""),
                    "negative_source": neg.get("source", ""),
                }
            )
    return pairs


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
