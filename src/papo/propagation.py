from __future__ import annotations

from statistics import pstdev
from typing import Any


def _merge_source(existing: str, incoming: str) -> str:
    tokens = []
    for value in [existing, incoming]:
        for token in str(value or "").split("+"):
            token = token.strip()
            if token and token not in tokens:
                tokens.append(token)
    return "+".join(tokens)


def candidate_rewards(tree: dict[str, Any]) -> list[dict[str, Any]]:
    if tree.get("leaves"):
        return leaf_propagated_rewards(tree)

    target = str(tree.get("target_action") or "")
    rows: list[dict[str, Any]] = []
    for node in tree.get("nodes", []):
        candidates = node.get("candidates", [])
        actions = [str(c.get("action") or "") for c in candidates]
        task_pass = {a: 1.0 for a in actions if a and a != "unknown"}
        user_pass = {a: 1.0 if a == target else 0.0 for a in actions}
        weights = {
            a: max(1.0, float(c.get("support", 1) or 1))
            for a, c in zip(actions, candidates)
        }
        for cand in candidates:
            action = str(cand.get("action") or "")
            rows.append(
                {
                    "tree_id": tree.get("tree_id", ""),
                    "node_id": node.get("node_id", ""),
                    "step_id": node.get("step_id", ""),
                    "user_id": node.get("user_id", ""),
                    "episode_id": node.get("episode_id", ""),
                    "action": action,
                    "source": cand.get("source", ""),
                    "support": cand.get("support", 1),
                    "leaf_weight": weights.get(action, 1.0),
                    "r_task": task_pass.get(action, 0.0),
                    "r_user": task_pass.get(action, 0.0) * user_pass.get(action, 0.0),
                }
            )
    return rows


def leaf_propagated_rewards(tree: dict[str, Any]) -> list[dict[str, Any]]:
    accum: dict[tuple[str, str], dict[str, Any]] = {}
    for leaf in tree.get("leaves", []):
        weight = float(leaf.get("leaf_weight", 1.0) or 1.0)
        r_user = float(leaf.get("r_user", 0.0) or 0.0)
        r_task = float(leaf.get("r_task", 0.0) or 0.0)
        for edge in leaf.get("path", []):
            node_id = str(edge.get("node_id") or "")
            action = str(edge.get("action") or "")
            key = (node_id, action)
            row = accum.setdefault(
                key,
                {
                    "tree_id": tree.get("tree_id", ""),
                    "node_id": node_id,
                    "step_id": tree.get("root_step_id", ""),
                    "user_id": tree.get("user_id", ""),
                    "episode_id": tree.get("episode_id", ""),
                    "action": action,
                    "source": edge.get("source", ""),
                    "support": 0.0,
                    "leaf_weight_sum": 0.0,
                    "weighted_r_user": 0.0,
                    "weighted_r_task": 0.0,
                    "num_leaf_paths": 0,
                },
            )
            row["support"] += float(edge.get("support", 1) or 1)
            row["leaf_weight_sum"] += weight
            row["weighted_r_user"] += weight * r_user
            row["weighted_r_task"] += weight * r_task
            row["num_leaf_paths"] += 1
            row["source"] = _merge_source(str(row.get("source", "")), str(edge.get("source", "")))

    rows: list[dict[str, Any]] = []
    for row in accum.values():
        denom = max(float(row.pop("leaf_weight_sum", 0.0) or 0.0), 1e-8)
        row["leaf_weight"] = denom
        row["r_user"] = float(row.pop("weighted_r_user")) / denom
        row["r_task"] = float(row.pop("weighted_r_task")) / denom
        rows.append(row)
    return rows


def add_advantages(rows: list[dict[str, Any]], alpha: float = 0.1) -> list[dict[str, Any]]:
    by_node: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("tree_id") or "") + "::" + str(row.get("node_id") or "")
        by_node.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for node_id, node_rows in by_node.items():
        if not node_rows:
            continue
        v_user = sum(float(r["r_user"]) for r in node_rows) / len(node_rows)
        v_task = sum(float(r["r_task"]) for r in node_rows) / len(node_rows)
        user_values = [float(r["r_user"]) for r in node_rows]
        spread = pstdev(user_values) if len(user_values) > 1 else 0.0
        for row in node_rows:
            row = dict(row)
            unc = 1.0 / ((float(row.get("support", 1) or 1) + 1.0) ** 0.5) + spread
            row["q_user"] = float(row["r_user"])
            row["q_task"] = float(row["r_task"])
            row["a_user"] = row["q_user"] - v_user
            row["a_task"] = row["q_task"] - v_task
            row["uncertainty"] = unc
            row["a_user_conservative"] = row["a_user"] - alpha * unc
            row["a_delta"] = row["a_user_conservative"] - row["a_task"]
            out.append(row)
    return out
