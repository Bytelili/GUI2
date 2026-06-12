from __future__ import annotations

from collections import defaultdict
from typing import Any

from .io import intent_key


def group_by_user(steps: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        grouped[str(step.get("user_id") or "")].append(step)
    for rows in grouped.values():
        rows.sort(
            key=lambda r: (
                int(r.get("chronological_rank", 0) or 0),
                str(r.get("time") or ""),
                str(r.get("episode_id") or ""),
                int(r.get("step_index") or 0),
            )
        )
    return dict(grouped)


def episode_order(steps: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    order: dict[str, tuple[str, str]] = {}
    for step in steps:
        episode_id = str(step.get("episode_id") or "")
        if episode_id and episode_id not in order:
            order[episode_id] = (str(step.get("user_id") or ""), str(step.get("time") or ""))
    return order


def similarity(query: dict[str, Any], candidate: dict[str, Any]) -> float:
    score = 0.0
    if intent_key(query) and intent_key(query) == intent_key(candidate):
        score += 0.6
    if str(query.get("app") or "") == str(candidate.get("app") or ""):
        score += 0.3
    if str(query.get("stage_label") or "") == str(candidate.get("stage_label") or ""):
        score += 0.1
    return score


def past_user_steps(
    query: dict[str, Any],
    all_steps: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    user_id = str(query.get("user_id") or "")
    q_rank = int(query.get("chronological_rank", 0) or 0)
    q_time = str(query.get("time") or "")
    q_episode = str(query.get("episode_id") or "")

    candidates = [
        s for s in all_steps
        if str(s.get("user_id") or "") == user_id
        and str(s.get("episode_id") or "") != q_episode
        and (
            int(s.get("chronological_rank", 0) or 0) < q_rank
            or (not q_rank and (not q_time or str(s.get("time") or "") < q_time))
        )
    ]
    candidates.sort(key=lambda s: (similarity(query, s), str(s.get("time") or "")), reverse=True)
    return candidates[:top_k]


def cross_user_steps(
    query: dict[str, Any],
    all_steps: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    user_id = str(query.get("user_id") or "")
    q_time = str(query.get("time") or "")
    candidates = [
        s for s in all_steps
        if str(s.get("user_id") or "") != user_id
        and (not q_time or str(s.get("time") or "") < q_time)
    ]
    candidates.sort(key=lambda s: similarity(query, s), reverse=True)
    return candidates[:top_k]
