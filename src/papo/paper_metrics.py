from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from statistics import mean
from typing import Any, Callable


def levenshtein_similarity(left: str | list[str], right: str | list[str]) -> float:
    left_items = list(left) if isinstance(left, str) else list(left)
    right_items = list(right) if isinstance(right, str) else list(right)
    if not left_items and not right_items:
        return 1.0
    if not left_items or not right_items:
        return 0.0

    previous = list(range(len(right_items) + 1))
    for i, left_item in enumerate(left_items, 1):
        current = [i]
        for j, right_item in enumerate(right_items, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    distance = previous[-1]
    return 1.0 - distance / max(len(left_items), len(right_items), 1)


def lexical_semantic_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left or "", right or "").ratio()


def proactive_metrics(
    rows: list[dict[str, Any]],
    semantic_similarity: Callable[[str, str], float] = lexical_semantic_similarity,
) -> dict[str, Any]:
    evaluated: list[dict[str, float]] = []
    for row in rows:
        target = str(row.get("target_intent") or row.get("original_intent") or "")
        prediction = str(row.get("predicted_intent") or row.get("prediction") or "")
        if not target or not prediction:
            continue
        s1 = float(semantic_similarity(target, prediction))
        s2 = levenshtein_similarity(target, prediction)
        success = row.get("success")
        evaluated.append(
            {
                "semantic_similarity_s1": s1,
                "levenshtein_similarity_s2": s2,
                "sim1": (s1 + s2) / 2,
                "success": float(_as_bool(success)) if success is not None else 0.0,
                "time": float(row.get("time", 0.0) or 0.0),
                "token": float(row.get("token", 0.0) or 0.0),
            }
        )
    return {
        "num_rows": len(rows),
        "num_evaluated": len(evaluated),
        "sim1": _avg(evaluated, "sim1"),
        "semantic_similarity_s1": _avg(evaluated, "semantic_similarity_s1"),
        "levenshtein_similarity_s2": _avg(evaluated, "levenshtein_similarity_s2"),
        "sr1": _avg(evaluated, "success") if any("success" in row for row in rows) else None,
        "avg_time": _avg(evaluated, "time"),
        "avg_token": _avg(evaluated, "token"),
        "notes": [
            "Paper Sim1 uses paraphrase-multilingual-MiniLM-L12-v2 cosine similarity for S1.",
            "This result uses the supplied semantic-similarity backend.",
            "SR1 requires a binary same-intent judge result in the success field.",
        ],
    }


def execution_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated: list[dict[str, float]] = []
    for row in rows:
        agent = _actions(row.get("agent_actions") or row.get("actions"))
        golden = _actions(row.get("golden_actions") or row.get("target_actions"))
        different_type = _actions(row.get("different_type_actions") or row.get("cross_user_actions"))
        if not agent or not golden:
            continue
        si = levenshtein_similarity(agent, golden)
        sii = levenshtein_similarity(agent, different_type) if different_type else 0.0
        success = _as_bool(row.get("success", False))
        over_limit = len(agent) > 2.5 * len(golden)
        success = success and not over_limit
        evaluated.append(
            {
                "success": float(success),
                "si": si,
                "sii": sii,
                "sim2": si / max(sii, 1e-8),
                "step_ratio": len(agent) / max(len(golden), 1),
                "successful_step_ratio": len(agent) / max(len(golden), 1) if success else 0.0,
                "time": float(row.get("time", 0.0) or 0.0),
                "token": float(row.get("token", 0.0) or 0.0),
            }
        )
    successful = [row for row in evaluated if row["success"] > 0]
    return {
        "num_rows": len(rows),
        "num_evaluated": len(evaluated),
        "sr2": _avg(evaluated, "success"),
        "sim2": _avg(evaluated, "sim2"),
        "si_user_similarity": _avg(evaluated, "si"),
        "sii_different_type_similarity": _avg(evaluated, "sii"),
        "step_ratio_success_only": _avg(successful, "successful_step_ratio"),
        "avg_time": _avg(evaluated, "time"),
        "avg_token": _avg(evaluated, "token"),
        "notes": [
            "Official SR2 requires final environment-state or manual success labels.",
            "Tasks above 2.5 times the golden steps are forced to failure.",
            "Different-type references should come from a different age group.",
        ],
    }


def papo_tree_proxy_metrics(trees: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    counterfactual_rows: list[dict[str, float]] = []
    source_counts: dict[str, int] = defaultdict(int)
    for tree in trees:
        leaves = tree.get("leaves") if isinstance(tree.get("leaves"), list) else []
        if not leaves:
            continue
        best = max(leaves, key=lambda leaf: (float(leaf.get("user_score", 0.0) or 0.0), float(leaf.get("r_task", 0.0) or 0.0)))
        actions = _actions(best.get("actions"))
        target = _actions(tree.get("target_actions"))
        for edge in best.get("path", []):
            for source in str(edge.get("source") or "").split("+"):
                if source:
                    source_counts[source] += 1
        rows.append(
            {
                "task_valid": float(best.get("r_task", 0.0) or 0.0),
                "user_valid": float(best.get("r_user", 0.0) or 0.0),
                "user_score": float(best.get("user_score", 0.0) or 0.0),
                "target_action_similarity": levenshtein_similarity(actions, target),
                "step_ratio": len(actions) / max(len(target), 1),
            }
        )
        counterfactual_leaves = [
            leaf for leaf in leaves
            if any(str(edge.get("source") or "") != "observed_path" for edge in leaf.get("path", []))
        ]
        if counterfactual_leaves:
            cf_best = max(
                counterfactual_leaves,
                key=lambda leaf: (float(leaf.get("user_score", 0.0) or 0.0), float(leaf.get("r_task", 0.0) or 0.0)),
            )
            cf_actions = _actions(cf_best.get("actions"))
            counterfactual_rows.append(
                {
                    "task_valid": float(cf_best.get("r_task", 0.0) or 0.0),
                    "user_valid": float(cf_best.get("r_user", 0.0) or 0.0),
                    "user_score": float(cf_best.get("user_score", 0.0) or 0.0),
                    "target_action_similarity": levenshtein_similarity(cf_actions, target),
                    "step_ratio": len(cf_actions) / max(len(target), 1),
                }
            )
    return {
        "num_trees": len(trees),
        "num_evaluated": len(rows),
        "best_leaf_task_valid_rate": _avg(rows, "task_valid"),
        "best_leaf_user_valid_rate": _avg(rows, "user_valid"),
        "best_leaf_avg_user_score": _avg(rows, "user_score"),
        "best_leaf_target_action_similarity": _avg(rows, "target_action_similarity"),
        "best_leaf_step_ratio": _avg(rows, "step_ratio"),
        "best_leaf_source_counts": dict(source_counts),
        "num_counterfactual_trees": len(counterfactual_rows),
        "best_counterfactual_task_valid_rate": _avg(counterfactual_rows, "task_valid"),
        "best_counterfactual_user_valid_rate": _avg(counterfactual_rows, "user_valid"),
        "best_counterfactual_avg_user_score": _avg(counterfactual_rows, "user_score"),
        "best_counterfactual_target_action_similarity": _avg(counterfactual_rows, "target_action_similarity"),
        "best_counterfactual_step_ratio": _avg(counterfactual_rows, "step_ratio"),
        "notes": [
            "These are offline PAPO proxy metrics, not paper SR2 or Sim2.",
            "Counterfactual metrics exclude leaves whose entire path is observed_path.",
            "Official SR2 requires online execution and final-state validation.",
        ],
    }


def execution_reference_metrics(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for task in tasks:
        inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
        target = task.get("target") if isinstance(task.get("target"), dict) else {}
        golden = _actions(target.get("actions"))
        same_ref = inputs.get("same_user_action_reference")
        cross_ref = inputs.get("cross_user_action_reference")
        same_actions = _actions(same_ref.get("actions")) if isinstance(same_ref, dict) else []
        cross_actions = _actions(cross_ref.get("actions")) if isinstance(cross_ref, dict) else []
        if not golden or not same_actions or not cross_actions:
            continue
        si = levenshtein_similarity(same_actions, golden)
        sii = levenshtein_similarity(same_actions, cross_actions)
        cross_to_target = levenshtein_similarity(cross_actions, golden)
        rows.append(
            {
                "same_user_to_target": si,
                "same_user_to_cross_user": sii,
                "cross_user_to_target": cross_to_target,
                "retrieval_sim2_proxy": si / max(sii, 1e-8),
                "personalization_gain": si - cross_to_target,
            }
        )
    return {
        "num_tasks": len(tasks),
        "num_evaluated": len(rows),
        "same_user_to_target_similarity": _avg(rows, "same_user_to_target"),
        "cross_user_to_target_similarity": _avg(rows, "cross_user_to_target"),
        "same_user_to_cross_user_similarity": _avg(rows, "same_user_to_cross_user"),
        "retrieval_sim2_proxy": _avg(rows, "retrieval_sim2_proxy"),
        "personalization_gain": _avg(rows, "personalization_gain"),
        "notes": [
            "This evaluates reference retrieval quality before an agent is run.",
            "It is not official Sim2 because the same-user reference is not an executed agent trajectory.",
        ],
    }


def suggestion_task_readiness(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for task in tasks:
        inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
        rows.append(
            {
                "has_profile": float(bool(inputs.get("user_profile"))),
                "has_history": float(bool(inputs.get("previous_intents"))),
                "num_history": float(len(inputs.get("previous_intents") or [])),
                "num_screenshots": float(len(inputs.get("initial_screenshots") or [])),
            }
        )
    return {
        "num_tasks": len(tasks),
        "profile_coverage": _avg(rows, "has_profile"),
        "history_coverage": _avg(rows, "has_history"),
        "avg_history_intents": _avg(rows, "num_history"),
        "avg_initial_screenshots": _avg(rows, "num_screenshots"),
    }


def _actions(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [item.strip() for item in value.split("，") if item.strip()]
    return []


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "success", "passed"}
    return bool(value)


def _avg(rows: list[dict[str, float]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    return mean(values) if values else None
