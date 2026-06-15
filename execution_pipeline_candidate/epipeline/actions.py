from __future__ import annotations

import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Any


ACTION_NAMES = {
    "click",
    "long_click",
    "type",
    "scroll",
    "press_back",
    "press_home",
    "press_recent",
    "wait",
    "finished",
}

# Literal U+FF0C separator used by the bound FingerTip-20K official source.
OFFICIAL_ACTION_SEPARATOR = "\uFF0C"


@dataclass(frozen=True)
class ParsedAction:
    raw: str
    action_type: str
    valid: bool
    x: int | None = None
    y: int | None = None
    direction: str = ""
    text: str = ""
    content: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "action_type": self.action_type,
            "valid": self.valid,
            "x": self.x,
            "y": self.y,
            "direction": self.direction,
            "text": self.text,
            "content": self.content,
            "error": self.error,
        }


def parse_action(text: str) -> ParsedAction:
    raw = str(text or "").strip()
    matches = re.findall(
        r"\b(click|long_click|type|scroll|press_back|press_home|press_recent|wait|finished)\b",
        raw,
        flags=re.IGNORECASE,
    )
    if len(matches) != 1:
        return ParsedAction(raw, "invalid", False, error="expected exactly one action")
    action_type = matches[0].lower()
    if action_type in {"press_back", "press_home", "press_recent", "wait", "finished"}:
        return ParsedAction(raw, action_type, True)
    if action_type == "type":
        typed = re.search(r"text\s*=\s*(['\"])(.*?)\1", raw, flags=re.DOTALL)
        if not typed:
            return ParsedAction(raw, action_type, False, error="type action is missing quoted text")
        return ParsedAction(raw, action_type, True, text=typed.group(2))

    coordinates = re.search(r"coordinates\s*=\s*\(([-\d]+)\s*,\s*([-\d]+)\)", raw)
    if not coordinates:
        return ParsedAction(raw, action_type, False, error="action is missing coordinates")
    x, y = int(coordinates.group(1)), int(coordinates.group(2))
    content_match = re.search(r"content\s*=\s*(['\"])(.*?)\1", raw, flags=re.DOTALL)
    content = content_match.group(2) if content_match else ""
    if action_type == "scroll":
        direction = re.search(r"\b(up|down|left|right)\b", raw, flags=re.IGNORECASE)
        if not direction:
            return ParsedAction(raw, action_type, False, x=x, y=y, error="scroll is missing direction")
        return ParsedAction(raw, action_type, True, x=x, y=y, direction=direction.group(1).lower())
    return ParsedAction(raw, action_type, True, x=x, y=y, content=content)


def levenshtein_similarity(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for index, left_item in enumerate(left, 1):
        current = [index]
        for other_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[other_index] + 1,
                    previous[other_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right), 1)


def official_action_text(actions: list[str], *, predicted: bool) -> str:
    """Match FingerTip's complete-action-text construction."""
    text = OFFICIAL_ACTION_SEPARATOR.join(str(value) for value in actions)
    return text + (OFFICIAL_ACTION_SEPARATOR if predicted and text else "")


def official_text_similarity(left: str, right: str) -> float:
    """Match personalized_execution.py: fuzz.ratio(...)/100 rounded to 2 decimals."""
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        ratio = int(round(100 * SequenceMatcher(None, str(left or ""), str(right or "")).ratio()))
    else:
        ratio = fuzz.ratio(str(left or ""), str(right or ""))
    return round(ratio / 100.0, 2)


def official_execution_similarity(
    agent_actions: list[str],
    golden_actions: list[str],
    cross_actions: list[str],
) -> tuple[float, float, float]:
    predicted = official_action_text(agent_actions, predicted=True)
    golden = official_action_text(golden_actions, predicted=False)
    cross = official_action_text(cross_actions, predicted=False)
    up_sim = official_text_similarity(golden, predicted)
    raw_down_sim = official_text_similarity(cross, predicted)
    down_sim = 0.4 if raw_down_sim == 0 else raw_down_sim
    return up_sim, down_sim, up_sim / down_sim
