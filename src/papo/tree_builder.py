from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .history import cross_user_steps, past_user_steps
from .io import action_label, state_key, step_id
from .schemas import PapoCandidate, PapoNode, PapoTree
from .verifier import DEFAULT_WEIGHTS, build_user_profile, score_leaf


@dataclass
class TreeBuildContext:
    all_steps: list[dict[str, Any]]
    by_id: dict[str, dict[str, Any]]
    by_episode: dict[str, list[dict[str, Any]]]
    by_user: dict[str, list[dict[str, Any]]]
    eligible_by_app: dict[str, list[dict[str, Any]]]
    eligible_by_state: dict[str, list[dict[str, Any]]]
    eligible_by_intent: dict[str, list[dict[str, Any]]]


def build_tree_context(all_steps: list[dict[str, Any]]) -> TreeBuildContext:
    by_id, by_episode = _episode_index(all_steps)
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_by_app: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_steps:
        by_user[str(row.get("user_id") or "")].append(row)
        if not row.get("has_next_state") or not row.get("valid_observation"):
            continue
        eligible_by_app[str(row.get("app") or "")].append(row)
        eligible_by_state[state_key(row)].append(row)
        intent = str(row.get("intent_key") or "")
        if intent:
            eligible_by_intent[intent].append(row)
    for rows in by_user.values():
        rows.sort(key=_rank)
    return TreeBuildContext(
        all_steps=all_steps,
        by_id=by_id,
        by_episode=by_episode,
        by_user=dict(by_user),
        eligible_by_app=dict(eligible_by_app),
        eligible_by_state=dict(eligible_by_state),
        eligible_by_intent=dict(eligible_by_intent),
    )


def merge_candidates(candidates: list[PapoCandidate], max_candidates: int) -> list[PapoCandidate]:
    merged: dict[str, PapoCandidate] = {}
    for cand in candidates:
        existing = merged.get(cand.action)
        if existing is None:
            merged[cand.action] = cand
            continue
        existing.support += cand.support
        existing.example_step_ids.extend(cand.example_step_ids)
        if cand.source not in existing.source.split("+"):
            existing.source = existing.source + "+" + cand.source
    out = list(merged.values())
    out.sort(key=lambda c: (c.action == "unknown", -c.support, c.source, c.action))
    return out[:max_candidates]


def build_depth1_tree(
    step: dict[str, Any],
    all_steps: list[dict[str, Any]],
    history_top_k: int = 5,
    same_user_k: int = 2,
    cross_user_k: int = 2,
    max_candidates: int = 6,
) -> PapoTree:
    target = action_label(step)
    history = past_user_steps(step, all_steps, top_k=history_top_k)
    same = past_user_steps(step, all_steps, top_k=same_user_k)
    cross = cross_user_steps(step, all_steps, top_k=cross_user_k)

    candidates: list[PapoCandidate] = [
        PapoCandidate(
            action=target,
            source="observed_leaf",
            support=1,
            example_step_ids=[step_id(step)],
        )
    ]

    for src_name, rows in [("same_user", same), ("cross_user", cross)]:
        support: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            support[action_label(row)].append(str(row.get("step_id") or ""))
        for label, ids in support.items():
            candidates.append(
                PapoCandidate(
                    action=label,
                    source=src_name,
                    support=len(ids),
                    valid=True,
                    example_step_ids=ids,
                )
            )

    node = PapoNode(
        node_id=f"papo_node__{step_id(step)}",
        step_id=step_id(step),
        user_id=str(step.get("user_id") or ""),
        episode_id=str(step.get("episode_id") or ""),
        step_index=int(step.get("step_index") or 0),
        intent=str(step.get("intent") or ""),
        app=str(step.get("app") or ""),
        state_key=state_key(step),
        history_ids=[str(h.get("episode_id") or "") for h in history],
        candidates=merge_candidates(candidates, max_candidates=max_candidates),
    )
    return PapoTree(
        tree_id=f"papo_tree__{step_id(step)}",
        root_step_id=step_id(step),
        target_action=target,
        nodes=[node],
        metadata={
            "tree_depth": 1,
            "history_top_k": history_top_k,
            "same_user_k": same_user_k,
            "cross_user_k": cross_user_k,
            "max_candidates": max_candidates,
            "note": "Depth-1 offline PAPO MVP tree.",
        },
    )


def _rank(step: dict[str, Any]) -> int:
    return int(step.get("chronological_rank", 0) or 0)


def _episode_index(all_steps: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_steps:
        sid = step_id(row)
        if sid:
            by_id[sid] = row
        by_episode[str(row.get("episode_id") or "")].append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda r: int(r.get("step_index", 0) or 0))
    return by_id, dict(by_episode)


def _next_step(row: dict[str, Any], by_episode: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    rows = by_episode.get(str(row.get("episode_id") or ""), [])
    idx = int(row.get("step_index", 0) or 0)
    if 0 <= idx + 1 < len(rows):
        return rows[idx + 1]
    return None


def _target_sequence(root: dict[str, Any], by_episode: dict[str, list[dict[str, Any]]], max_depth: int) -> list[str]:
    rows = by_episode.get(str(root.get("episode_id") or ""), [])
    idx = int(root.get("step_index", 0) or 0)
    return [action_label(r) for r in rows[idx : idx + max_depth]]


def _token_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    left = set(str(x) for x in (a.get("object_tokens") or []))
    right = set(str(x) for x in (b.get("object_tokens") or []))
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _similarity(query: dict[str, Any], cand: dict[str, Any]) -> float:
    score = 0.0
    if str(query.get("app") or "") == str(cand.get("app") or ""):
        score += 2.0
    if state_key(query) == state_key(cand):
        score += 3.0
    if str(query.get("intent_key") or "") and str(query.get("intent_key") or "") == str(cand.get("intent_key") or ""):
        score += 1.0
    score += _token_overlap(query, cand)
    return score


def _candidate_rows(
    query_step: dict[str, Any],
    root_step: dict[str, Any],
    context: TreeBuildContext,
    same_user_k: int,
    cross_user_k: int,
) -> list[tuple[str, dict[str, Any]]]:
    root_user = str(root_step.get("user_id") or "")
    root_rank = _rank(root_step)
    root_time = str(root_step.get("time") or "")
    rows: list[tuple[str, dict[str, Any], float]] = []

    pool: dict[str, dict[str, Any]] = {}
    for rows_for_key in [
        context.eligible_by_app.get(str(query_step.get("app") or ""), []),
        context.eligible_by_state.get(state_key(query_step), []),
        context.eligible_by_intent.get(str(query_step.get("intent_key") or ""), []),
    ]:
        for row in rows_for_key:
            pool[step_id(row)] = row

    for row in pool.values():
        if step_id(row) == step_id(query_step):
            continue
        sim = _similarity(query_step, row)
        if sim <= 0:
            continue
        user = str(row.get("user_id") or "")
        rank = _rank(row)
        if user == root_user and rank < root_rank:
            rows.append(("same_user", row, sim + 0.25))
        elif user != root_user and (not root_time or str(row.get("time") or "") < root_time):
            rows.append(("cross_user", row, sim))

    same = sorted([r for r in rows if r[0] == "same_user"], key=lambda x: (x[2], _rank(x[1])), reverse=True)[:same_user_k]
    cross = sorted([r for r in rows if r[0] == "cross_user"], key=lambda x: (x[2], _rank(x[1])), reverse=True)[:cross_user_k]
    return [(src, row) for src, row, _sim in same + cross]


def _merge_transition_candidates(
    observed_step: dict[str, Any],
    retrieved: list[tuple[str, dict[str, Any]]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}

    def add(src: str, row: dict[str, Any]) -> None:
        label = action_label(row)
        if not label:
            return
        item = buckets.setdefault(
            label,
            {
                "action": label,
                "source": src,
                "support": 0,
                "transition_step_ids": [],
                "example_step_id": step_id(row),
                "valid": bool(row.get("has_next_state")),
            },
        )
        item["support"] += 1
        item["transition_step_ids"].append(step_id(row))
        if src not in str(item["source"]).split("+"):
            item["source"] = str(item["source"]) + "+" + src

    add("observed_path", observed_step)
    for src, row in retrieved:
        add(src, row)

    out = list(buckets.values())
    out.sort(key=lambda c: ("observed_path" not in str(c["source"]), -int(c["support"]), str(c["action"])))
    return out[:max_candidates]


def build_offline_counterfactual_tree(
    root_step: dict[str, Any],
    all_steps: list[dict[str, Any]],
    max_depth: int = 3,
    same_user_k: int = 2,
    cross_user_k: int = 2,
    max_candidates: int = 4,
    user_threshold: float = 0.6,
    verifier_weights: dict[str, float] | None = None,
    context: TreeBuildContext | None = None,
) -> dict[str, Any]:
    context = context or build_tree_context(all_steps)
    by_id, by_episode = context.by_id, context.by_episode
    target_seq = _target_sequence(root_step, by_episode, max_depth=max_depth)
    user_profile = build_user_profile(root_step, context.by_user.get(str(root_step.get("user_id") or ""), []))
    weights = verifier_weights or DEFAULT_WEIGHTS
    nodes: list[dict[str, Any]] = []
    leaves: list[dict[str, Any]] = []
    node_counter = 0

    def recurse(proxy_step: dict[str, Any], depth: int, path: list[dict[str, Any]]) -> None:
        nonlocal node_counter
        node_id = f"node_{node_counter:05d}"
        node_counter += 1

        retrieved = _candidate_rows(
            proxy_step,
            root_step,
            context,
            same_user_k=same_user_k,
            cross_user_k=cross_user_k,
        )
        candidates = _merge_transition_candidates(proxy_step, retrieved, max_candidates=max_candidates)
        nodes.append(
            {
                "node_id": node_id,
                "proxy_step_id": step_id(proxy_step),
                "depth": depth,
                "state_key": state_key(proxy_step),
                "prefix_actions": [p["action"] for p in path],
                "candidates": candidates,
            }
        )

        if depth >= max_depth or not candidates:
            actions = [p["action"] for p in path]
            leaves.append(_make_leaf(root_step, target_seq, actions, path, user_profile, user_threshold, weights))
            return

        for cand in candidates:
            transition = by_id.get(str(cand.get("example_step_id") or ""))
            if transition is None:
                continue
            next_proxy = _next_step(transition, by_episode)
            new_path = path + [
                {
                    "node_id": node_id,
                    "action": cand["action"],
                    "source": cand["source"],
                    "support": cand["support"],
                    "transition_step_id": step_id(transition),
                }
            ]
            if next_proxy is None or depth + 1 >= max_depth:
                leaves.append(_make_leaf(root_step, target_seq, [p["action"] for p in new_path], new_path, user_profile, user_threshold, weights))
            else:
                recurse(next_proxy, depth + 1, new_path)

    recurse(root_step, 0, [])
    return {
        "tree_id": f"papo_tree__{step_id(root_step)}",
        "root_step_id": step_id(root_step),
        "user_id": str(root_step.get("user_id") or ""),
        "episode_id": str(root_step.get("episode_id") or ""),
        "target_actions": target_seq,
        "nodes": nodes,
        "leaves": leaves,
        "metadata": {
            "tree_type": "offline_counterfactual_transition_tree",
            "max_depth": max_depth,
            "same_user_k": same_user_k,
            "cross_user_k": cross_user_k,
            "max_candidates": max_candidates,
            "uses_raw_papo_steps": True,
            "user_threshold": user_threshold,
            "verifier_weights": weights,
            "user_history_steps": user_profile.get("num_history_steps", 0),
        },
    }


def _make_leaf(
    root_step: dict[str, Any],
    target_seq: list[str],
    actions: list[str],
    path: list[dict[str, Any]],
    user_profile: dict[str, Any],
    user_threshold: float,
    verifier_weights: dict[str, float],
) -> dict[str, Any]:
    prefix_target = target_seq[: len(actions)]
    prefix_matches = sum(1 for a, b in zip(actions, prefix_target) if a == b)
    verifier = score_leaf(actions, target_seq, path, user_profile, verifier_weights)
    user_pass = bool(actions and verifier["total"] >= user_threshold)
    task_pass = bool(actions and all(a and a != "unknown" for a in actions))
    return {
        "leaf_id": f"leaf_{stable_leaf_id(actions, path)}",
        "root_step_id": step_id(root_step),
        "actions": actions,
        "path": path,
        "target_actions": target_seq,
        "prefix_match_rate": prefix_matches / max(len(actions), 1),
        "user_score": verifier["total"],
        "user_score_components": verifier,
        "r_task": 1.0 if task_pass else 0.0,
        "r_user": 1.0 if task_pass and user_pass else 0.0,
        "leaf_weight": max(1.0, sum(float(p.get("support", 1) or 1) for p in path) / max(len(path), 1)),
    }


def stable_leaf_id(actions: list[str], path: list[dict[str, Any]]) -> str:
    import hashlib

    raw = "||".join(actions) + "::" + "||".join(str(p.get("transition_step_id", "")) for p in path)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
