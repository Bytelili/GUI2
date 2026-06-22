from __future__ import annotations

import re

from tn_dpo_gui.data.action_normalizer import deduplicate_actions
from tn_dpo_gui.data.action_schema import Action

ROLE_PATTERN = re.compile(r"(button|clickable|link|tab|checkbox|input|search|textbox)", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"([A-Za-z0-9][A-Za-z0-9 _-]{1,40})")


def extract_affordance_actions(ui_tree: str | None, max_actions: int = 6) -> list[Action]:
    if not ui_tree:
        return []
    candidates: list[Action] = []
    for raw_line in ui_tree.splitlines():
        line = raw_line.strip()
        if not line or not ROLE_PATTERN.search(line):
            continue
        token_match = TOKEN_PATTERN.search(line)
        target = token_match.group(1).strip() if token_match else line[:40]
        lowered = line.lower()
        if any(keyword in lowered for keyword in ["input", "textbox", "search"]):
            candidates.append(Action(action_type="type", target=target, text="example query"))
        else:
            candidates.append(Action(action_type="click", target=target))
        if "scroll" in lowered:
            candidates.append(Action(action_type="scroll", text="down"))
        if len(candidates) >= max_actions:
            break
    return deduplicate_actions(candidates)[:max_actions]
