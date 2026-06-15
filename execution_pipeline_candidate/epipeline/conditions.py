from __future__ import annotations

import copy
import random
from typing import Any


CONDITIONS = {
    "correct_full_history",
    "correct_recent_history",
    "no_history",
    "cross_user_history",
    "shuffled_user_history",
    "stale_history",
    "truncated_history",
}

OFFICIAL_AGE_GROUP_2 = {
    "30", "70", "73", "74", "75", "77", "79", "80",
    "86", "88", "89", "93", "94", "95", "96", "97",
}


def apply_condition(
    tasks: list[dict[str, Any]],
    condition: str,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown execution condition: {condition}")
    validate_source_tasks(tasks)
    donors = [
        copy.deepcopy(reference)
        for task in tasks
        for reference in same_references(task)
    ]
    output: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for task in tasks:
        result, reason = transform_task(task, condition, donors=donors, seed=seed)
        if result is None:
            excluded.append({"task_id": task_id(task), "reason": reason})
        else:
            output.append(result)
    manifest = {
        "condition": condition,
        "source_tasks": len(tasks),
        "eligible_tasks": len(output),
        "excluded_tasks": len(excluded),
        "excluded": excluded,
    }
    return output, manifest


def transform_task(
    task: dict[str, Any],
    condition: str,
    *,
    donors: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any] | None, str]:
    same = same_references(task)
    cross = cross_references(task)
    if condition == "correct_full_history":
        if not same:
            return None, "missing_same_user_history"
        return marked(task, condition), ""
    if condition == "correct_recent_history":
        if not same:
            return None, "missing_same_user_history"
        return set_same_references(task, [latest(same)], condition), ""
    if condition == "truncated_history":
        if len(same) < 2:
            return None, "fewer_than_two_same_user_references"
        ordered = sorted(same, key=lambda item: str(item.get("time") or ""), reverse=True)
        keep = max(1, (len(ordered) + 1) // 2)
        return set_same_references(task, ordered[:keep], condition), ""
    if condition == "no_history":
        result = set_same_references(task, [], condition)
        inputs(result)["cross_user_action_references"] = []
        inputs(result)["cross_user_action_reference"] = None
        return result, ""
    if condition == "cross_user_history":
        if not cross:
            return None, "missing_cross_user_history"
        return set_same_references(task, [latest(cross)], condition), ""
    if condition == "stale_history":
        if len(same) < 2:
            return None, "fewer_than_two_same_user_references"
        return set_same_references(task, [earliest(same)], condition), ""
    if condition == "shuffled_user_history":
        current = inputs(task)
        user_id = str(current.get("user_id") or "")
        target_time = str(current.get("time") or "")
        eligible = [
            donor
            for donor in donors
            if str(donor.get("user_id") or "") != user_id
            and str(donor.get("time") or "") < target_time
        ]
        if not eligible:
            return None, "missing_strictly_earlier_cross_user_donor"
        rng = random.Random(f"{seed}:{task_id(task)}:{condition}")
        return set_same_references(task, [eligible[rng.randrange(len(eligible))]], condition), ""
    raise AssertionError(condition)


def validate_source_tasks(tasks: list[dict[str, Any]]) -> None:
    identifiers = [task_id(task) for task in tasks]
    if not tasks or any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Execution tasks must be non-empty with unique task IDs")
    for task in tasks:
        current = inputs(task)
        target = task.get("target") if isinstance(task.get("target"), dict) else {}
        target_time = str(current.get("time") or "")
        if not current.get("user_id") or not target_time or not current.get("instruction"):
            raise ValueError(f"Incomplete execution task input: {task_id(task)}")
        if not target.get("actions"):
            raise ValueError(f"Execution task has no golden actions: {task_id(task)}")
        for reference in same_references(task):
            if str(reference.get("user_id") or "") != str(current.get("user_id") or ""):
                raise ValueError(f"Same-user identity mismatch: {task_id(task)}")
            if str(reference.get("time") or "") >= target_time:
                raise ValueError(f"Non-temporal same-user reference: {task_id(task)}")
        for reference in cross_references(task):
            if str(reference.get("user_id") or "") == str(current.get("user_id") or ""):
                raise ValueError(f"Cross-user identity mismatch: {task_id(task)}")
            if official_age_group(reference.get("user_id")) == official_age_group(current.get("user_id")):
                raise ValueError(f"Cross-user reference is not from the official different type: {task_id(task)}")
            if str(reference.get("time") or "") >= target_time:
                raise ValueError(f"Non-temporal cross-user reference: {task_id(task)}")
        if not cross_references(task):
            raise ValueError(f"Execution task has no official different-type reference: {task_id(task)}")
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        if metadata.get("cross_user_reference_is_different_age_group_counterfactual") is not True:
            raise ValueError(f"Execution task lacks different-type provenance: {task_id(task)}")


def marked(task: dict[str, Any], condition: str) -> dict[str, Any]:
    result = copy.deepcopy(task)
    metadata = result.setdefault("metadata", {})
    cross = cross_references(result)
    if cross:
        metadata["evaluation_cross_user_actions"] = copy.deepcopy(cross[0].get("actions") or [])
        metadata["evaluation_cross_user_id"] = str(cross[0].get("user_id") or "")
        metadata["evaluation_cross_user_reference_hidden_from_model"] = True
    metadata["execution_pipeline_condition"] = condition
    metadata["target_actions_hidden_from_model"] = True
    return result


def set_same_references(task: dict[str, Any], references: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    result = marked(task, condition)
    current = inputs(result)
    current["same_user_action_references"] = copy.deepcopy(references)
    current["same_user_action_reference"] = copy.deepcopy(references[0]) if references else None
    return result


def task_id(task: dict[str, Any]) -> str:
    return str(task.get("task_id") or "")


def inputs(task: dict[str, Any]) -> dict[str, Any]:
    value = task.get("input")
    if not isinstance(value, dict):
        value = {}
        task["input"] = value
    return value


def same_references(task: dict[str, Any]) -> list[dict[str, Any]]:
    value = inputs(task).get("same_user_action_references")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def cross_references(task: dict[str, Any]) -> list[dict[str, Any]]:
    value = inputs(task).get("cross_user_action_references")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def latest(references: list[dict[str, Any]]) -> dict[str, Any]:
    return max(references, key=lambda item: str(item.get("time") or ""))


def earliest(references: list[dict[str, Any]]) -> dict[str, Any]:
    return min(references, key=lambda item: str(item.get("time") or ""))


def official_age_group(user_id: Any) -> str:
    return "group_2" if str(user_id) in OFFICIAL_AGE_GROUP_2 else "group_1"
