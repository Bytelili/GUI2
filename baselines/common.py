from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path
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


def sequence_similarity(left: list[str] | str, right: list[str] | str) -> float:
    """Match the official implementation's fuzzy whole-sequence comparison."""
    left_text = "，".join(left) if isinstance(left, list) else str(left or "")
    right_text = "，".join(right) if isinstance(right, list) else str(right or "")
    return round(SequenceMatcher(None, left_text, right_text).ratio(), 4)


def levenshtein_similarity(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i]
        for j, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right), 1)


def parse_action(text: str) -> dict[str, Any]:
    """Parse an official FingerTip action without silently replacing failures by wait."""
    matches = re.findall(
        r"\b(click|long_click|type|scroll|press_back|press_home|press_recent|wait|finished)\b",
        text or "",
        flags=re.IGNORECASE,
    )
    if len(matches) != 1:
        return {"valid": False, "action_type": "invalid", "raw": text}
    action_type = matches[0].lower()
    result: dict[str, Any] = {"valid": True, "action_type": action_type, "raw": text}
    if action_type in {"press_back", "press_home", "press_recent", "wait", "finished"}:
        return result
    if action_type in {"click", "long_click", "scroll"}:
        coordinates = re.search(r"coordinates\s*=\s*\(([-\d]+)\s*,\s*([-\d]+)\)", text)
        if not coordinates:
            result["valid"] = False
            return result
        result["coordinates"] = [int(coordinates.group(1)), int(coordinates.group(2))]
    if action_type == "scroll":
        direction = re.search(r"\b(up|down|left|right)\b", text, flags=re.IGNORECASE)
        if not direction:
            result["valid"] = False
            return result
        result["direction"] = direction.group(1).lower()
    if action_type == "type":
        typed = re.search(r"text\s*=\s*(['\"])(.*?)\1", text)
        if not typed:
            result["valid"] = False
            return result
        result["text"] = typed.group(2)
    return result


def action_type(text: str) -> str:
    return str(parse_action(text).get("action_type") or "invalid")


def official_screen_description(xml_path: str | Path) -> str:
    """Reproduce the official XML-to-screen-description representation."""
    path = Path(xml_path)
    if not path.exists():
        return "[]"
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return "[]"
    nodes: list[tuple[str, tuple[int, int]]] = []
    for node in root.iter("node"):
        values = [
            str(node.get(name, "") or "").strip()
            for name in ("text", "content_desc", "hint")
        ]
        content = "，".join(value for value in values if value)
        bounds = str(node.get("bounds", "") or "")
        coords = re.findall(r"-?\d+", bounds)
        if not content or len(coords) != 4:
            continue
        x1, y1, x2, y2 = map(int, coords)
        nodes.append((content, ((x1 + x2) // 2, (y1 + y2) // 2)))
    return str(nodes)


def profile_text(profile: dict[str, Any]) -> str:
    values = [str(value) for value in profile.values() if str(value).strip()]
    return "，".join(values)


def official_prompt(
    instruction: str,
    profile: str,
    size: str,
    screen_description: str,
    actions_reference: list[str] | str,
    previous_actions: list[str] | str,
) -> str:
    reference_text = "，".join(actions_reference) if isinstance(actions_reference, list) else actions_reference
    previous_text = "，".join(previous_actions) if isinstance(previous_actions, list) else previous_actions
    return f"""You are an Android GUI agent. You are given an instruction and current screenshot and some supplementary information. You need to perform the next action to complete the instruction.

## Input
User_instruction: {instruction}
User_profile: {profile}
Screen_width_height: {size}
Screen_description: {screen_description}
Actions_reference: {reference_text}
Previous_actions: {previous_text}

## Action Space
click(coordinates=(x,y), content='')
long_click(coordinates=(x,y), content='')
type(text='')
scroll(coordinates=(x,y), direction='down or up or right or left')
press_back()
press_home()
press_recent()
wait()
finished()

## Note
- 'coordinates' should represent the coordinates of the click point. The origin is the upper left corner of the screenshot, with x increasing to the right and y increasing downward.
- 'content' should represent the original text at the click point or the description of the icon, usually in Chinese.
- 'text' should represent all the original text that the user intends to input.
- 'press_back()', 'press_home()', 'press_recent()' means going to the previous screen, home screen, or recent apps screen.
- 'wait()' means waiting until the next observation is received.
- 'finished()' means that the instruction is completed.
- Screen_description contains some correct 'content' and 'coordinates' of the UI, which can be directly referenced.
- Actions_reference represents the complete sequence of actions that the user performed when executing a similar instruction in the past.
- Previous_actions contains the sequence of actions already performed under the current instruction.
- Only one action in Action Space can be taken. Do not output anything other than the action to take.

The action to take:
"""


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

