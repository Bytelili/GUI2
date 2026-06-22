from __future__ import annotations

from typing import Iterable

from .action_schema import Action


def normalize_action(value: Action | dict | str) -> Action:
    if isinstance(value, Action):
        return value
    if isinstance(value, str):
        return Action(action_type="freeform", text=value)
    if isinstance(value, dict):
        return Action.from_dict(value)
    raise TypeError(f"Unsupported action payload: {type(value)!r}")


def deduplicate_actions(actions: Iterable[Action | dict | str]) -> list[Action]:
    deduped: list[Action] = []
    seen: set[str] = set()
    for action_like in actions:
        action = normalize_action(action_like)
        key = action.normalized_key()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped


def action_texts(actions: Iterable[Action | dict | str]) -> list[str]:
    return [normalize_action(action_like).to_text() for action_like in actions]
