from __future__ import annotations

import math
from collections import defaultdict
from statistics import pstdev
from typing import Any

from .paper_metrics import levenshtein_similarity


def score_tree_leaves(
    tree: dict[str, Any],
    task: dict[str, Any] | None,
    reward_config: dict[str, Any],
) -> dict[str, Any]:
    same_actions, cross_actions = _reference_actions(task)
    target_actions = [str(action) for action in tree.get("target_actions", [])]
    min_refs = int(reward_config.get("min_same_user_references", 1) or 1)
    eta = float(reward_config.get("eta", 0.5) or 0.5)
    temperature = max(float(reward_config.get("fingertip_temperature", 0.2) or 0.2), 1e-8)
    epsilon = max(float(reward_config.get("epsilon", 1e-6) or 1e-6), 1e-12)
    evidence_transform = str(reward_config.get("evidence_transform", "tanh_log_ratio") or "tanh_log_ratio")
    threshold = float(reward_config.get("task_similarity_threshold", 0.5) or 0.5)
    require_finish = bool(reward_config.get("require_finish", False))

    scored = dict(tree)
    scored_leaves: list[dict[str, Any]] = []
    for leaf in tree.get("leaves", []):
        actions = [str(action) for action in leaf.get("actions", [])]
        target_similarity = levenshtein_similarity(actions, target_actions)
        has_finish = any(action in {"finished", "finish", "FINISH()"} for action in actions)
        r_task = float(target_similarity >= threshold and (has_finish or not require_finish))
        s_positive = _max_similarity(actions, same_actions)
        s_negative = _max_similarity(actions, cross_actions)
        active = len(same_actions) >= min_refs
        residual_log_ratio = math.log((s_positive + epsilon) / (s_negative + epsilon)) if active else 0.0
        r_pref = _personalization_evidence(residual_log_ratio, temperature, evidence_transform) if active else 0.0
        r_user = r_task * (1.0 + eta * r_pref)
        item = dict(leaf)
        item.update(
            {
                "r_task": r_task,
                "r_pref": r_pref,
                "r_user": r_user,
                "task_similarity": target_similarity,
                "same_user_similarity": s_positive,
                "cross_user_similarity": s_negative,
                "residual_log_ratio": residual_log_ratio,
                "personalization_active": active,
            }
        )
        scored_leaves.append(item)
    scored["leaves"] = scored_leaves
    scored.setdefault("metadata", {})["objective"] = "papo_residual_outcome_propagation"
    scored["metadata"]["personalization_evidence_transform"] = evidence_transform
    return scored


def propagate_residual_values(
    trees: list[dict[str, Any]],
    alpha: float,
    beta: float,
    coverage_kappa: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    action_rows: list[dict[str, Any]] = []
    listwise_rows: list[dict[str, Any]] = []
    for tree in trees:
        node_context = {
            str(node.get("node_id") or ""): {
                "step_id": str(node.get("proxy_step_id") or tree.get("root_step_id") or ""),
                "prefix_actions": list(node.get("prefix_actions") or []),
                "state_key": str(node.get("state_key") or ""),
            }
            for node in tree.get("nodes", [])
        }
        by_action: dict[tuple[str, str], dict[str, Any]] = {}
        for leaf in tree.get("leaves", []):
            weight = max(float(leaf.get("leaf_weight", 1.0) or 1.0), 1e-8)
            for edge in leaf.get("path", []):
                key = (str(edge.get("node_id") or ""), str(edge.get("action") or ""))
                item = by_action.setdefault(
                    key,
                    {
                        "tree_id": tree.get("tree_id", ""),
                        "episode_id": tree.get("episode_id", ""),
                        "user_id": tree.get("user_id", ""),
                        "step_id": node_context.get(key[0], {}).get("step_id", tree.get("root_step_id", "")),
                        "node_id": key[0],
                        "prefix_actions": node_context.get(key[0], {}).get("prefix_actions", []),
                        "state_key": node_context.get(key[0], {}).get("state_key", ""),
                        "action": key[1],
                        "source": edge.get("source", ""),
                        "support": 0.0,
                        "weight_sum": 0.0,
                        "r_user_values": [],
                        "weighted_r_user": 0.0,
                        "weighted_r_task": 0.0,
                        "weighted_r_pref": 0.0,
                    },
                )
                item["support"] += float(edge.get("support", 1) or 1)
                item["weight_sum"] += weight
                item["weighted_r_user"] += weight * float(leaf.get("r_user", 0.0) or 0.0)
                item["weighted_r_task"] += weight * float(leaf.get("r_task", 0.0) or 0.0)
                item["weighted_r_pref"] += weight * float(leaf.get("r_pref", 0.0) or 0.0)
                item["r_user_values"].append(float(leaf.get("r_user", 0.0) or 0.0))

        by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in by_action.values():
            denom = max(float(item.pop("weight_sum")), 1e-8)
            values = item.pop("r_user_values")
            item["q_user"] = float(item.pop("weighted_r_user")) / denom
            item["q_task"] = float(item.pop("weighted_r_task")) / denom
            item["q_pref"] = float(item.pop("weighted_r_pref")) / denom
            uncertainty = 1.0 / math.sqrt(float(item["support"]) + 1.0)
            uncertainty += pstdev(values) if len(values) > 1 else 0.0
            item["uncertainty"] = uncertainty
            item["q_user_conservative"] = item["q_user"] - alpha * uncertainty
            item["a_delta"] = item["q_user_conservative"] - item["q_task"]
            item["coverage"] = 1.0 - math.exp(-float(item["support"]) / max(coverage_kappa, 1e-8))
            by_node[str(item["node_id"])].append(item)
            action_rows.append(item)

        for node_id, candidates in by_node.items():
            priors = _support_priors(candidates)
            logits = [
                math.log(max(prior, 1e-12)) + float(candidate["a_delta"]) / max(beta, 1e-8)
                for prior, candidate in zip(priors, candidates)
            ]
            target_policy = _softmax(logits)
            listwise_rows.append(
                {
                    "tree_id": tree.get("tree_id", ""),
                    "episode_id": tree.get("episode_id", ""),
                    "user_id": tree.get("user_id", ""),
                    "step_id": node_context.get(node_id, {}).get("step_id", tree.get("root_step_id", "")),
                    "node_id": node_id,
                    "prefix_actions": node_context.get(node_id, {}).get("prefix_actions", []),
                    "state_key": node_context.get(node_id, {}).get("state_key", ""),
                    "beta": beta,
                    "candidates": [
                        {
                            **candidate,
                            "base_policy_probability": prior,
                            "target_policy_probability": probability,
                        }
                        for candidate, prior, probability in zip(candidates, priors, target_policy)
                    ],
                }
            )
    return action_rows, listwise_rows


def _reference_actions(task: dict[str, Any] | None) -> tuple[list[list[str]], list[list[str]]]:
    if not task:
        return [], []
    inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
    same_refs = inputs.get("same_user_action_references") or [inputs.get("same_user_action_reference")]
    cross_refs = inputs.get("cross_user_action_references") or [inputs.get("cross_user_action_reference")]
    same_actions = [
        list(map(str, ref.get("actions", [])))
        for ref in same_refs
        if isinstance(ref, dict) and ref.get("actions")
    ]
    cross_actions = [
        list(map(str, ref.get("actions", [])))
        for ref in cross_refs
        if isinstance(ref, dict) and ref.get("actions")
    ]
    return same_actions, cross_actions


def _max_similarity(actions: list[str], references: list[list[str]]) -> float:
    return max((levenshtein_similarity(actions, reference) for reference in references), default=0.0)


def _support_priors(candidates: list[dict[str, Any]]) -> list[float]:
    values = [max(float(candidate.get("support", 1.0) or 1.0), 1e-8) for candidate in candidates]
    total = sum(values)
    return [value / total for value in values]


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exp_values = [math.exp(value - maximum) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def _personalization_evidence(log_ratio: float, temperature: float, transform: str) -> float:
    scaled = log_ratio / max(temperature, 1e-8)
    if transform == "tanh_log_ratio":
        return math.tanh(scaled)
    if transform == "sigmoid_log_ratio":
        return _sigmoid(scaled)
    raise ValueError(
        f"Unknown personalization evidence transform: {transform!r}. "
        "Expected 'tanh_log_ratio' or 'sigmoid_log_ratio'."
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
