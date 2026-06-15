from __future__ import annotations

import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def image_size(path: str | Path) -> str:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return f"{image.size[0]}x{image.size[1]}"
    except Exception:
        return "unknown"


def screen_description(xml_path: str | Path) -> str:
    path = Path(xml_path)
    if not path.exists():
        return "[]"
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return "[]"
    nodes: list[tuple[str, tuple[int, int]]] = []
    for node in root.iter("node"):
        content = "，".join(
            value
            for name in ("text", "content-desc", "content_desc", "hint")
            if (value := str(node.get(name, "") or "").strip())
        )
        coordinates = re.findall(r"-?\d+", str(node.get("bounds", "") or ""))
        if not content or len(coordinates) != 4:
            continue
        x1, y1, x2, y2 = map(int, coordinates)
        nodes.append((content, ((x1 + x2) // 2, (y1 + y2) // 2)))
    return str(nodes)


def profile_text(profile: dict[str, Any]) -> str:
    fields = {
        "sex": str(profile.get("sex") or ""),
        "age": str(profile.get("age") or ""),
        "occupation": str(profile.get("occupation") or ""),
        "address": str(profile.get("address") or ""),
        "family": str(profile.get("family") or ""),
        "phone_brand": str(profile.get("phone_brand") or profile.get("phone") or ""),
    }
    if any(fields.values()):
        return (
            f"{fields['sex']}，{fields['age']}岁，职业为{fields['occupation']}，"
            f"现居{fields['address']}，{fields['family']}，使用{fields['phone_brand']}手机"
        )
    return " | ".join(str(value) for value in profile.values() if str(value).strip())


SYSTEM_PROMPT = (
    "You are a personalized Android GUI agent. Follow the requested output format exactly. "
    "Use user history only when it is relevant and never reveal hidden target fields."
)


def object_tokens(xml_path: str | Path, max_nodes: int = 48) -> list[str]:
    path = Path(xml_path)
    if not path.exists():
        return []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    output: list[str] = []
    for node in list(root.iter("node"))[:max_nodes]:
        text = " ".join(
            value
            for name in ("text", "content-desc", "content_desc", "hint")
            if (value := str(node.get(name, "") or "").strip())
        )[:32]
        class_name = str(node.get("class", "") or "")
        view_id = str(node.get("resource-id", "") or node.get("view_id", "") or "")[-32:]
        bounds = re.findall(r"-?\d+", str(node.get("bounds", "") or ""))
        region = "unknown"
        if len(bounds) == 4:
            x1, y1, x2, y2 = map(int, bounds)
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            vertical = "top" if y < 600 else "bottom" if y > 1800 else "middle"
            horizontal = "left" if x < 360 else "right" if x > 720 else "center"
            region = "center" if vertical == "middle" and horizontal == "center" else f"{vertical}_{horizontal}"
        if text or node.get("clickable") == "true" or node.get("editable") == "true" or node.get("scrollable") == "true":
            output.append("|".join([class_name, text, view_id, region]))
    return output


def build_prompt(
    task: dict[str, Any],
    screenshot: str,
    xml_path: str,
    previous_actions: list[str],
    *,
    style: str = "training_aligned",
) -> str:
    if style == "official_reference":
        return build_official_reference_prompt(task, screenshot, xml_path, previous_actions)
    if style != "training_aligned":
        raise ValueError(f"Unsupported prompt style: {style}")
    inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
    references = inputs.get("same_user_action_references")
    references = references if isinstance(references, list) else []
    reference_actions = [
        reference.get("actions", [])
        for reference in references
        if isinstance(reference, dict)
    ]
    prompt = "\n".join(
        [
            "Predict exactly one next Android action and output no explanation.",
            f"Instruction: {inputs.get('instruction', '')}",
            f"Scenario: {inputs.get('scenario', '')}",
            f"User profile: {json.dumps(inputs.get('user_profile', {}), ensure_ascii=False)}",
            f"Relevant same-user reference actions: {json.dumps(reference_actions, ensure_ascii=False)}",
            f"Previous actions: {json.dumps(previous_actions, ensure_ascii=False)}",
            f"Current UI elements: {json.dumps(object_tokens(xml_path), ensure_ascii=False)}",
            "Allowed actions: click, long_click, type, scroll, press_back, press_home, press_recent, wait, finished.",
        ]
    )
    return ("<image>" if screenshot else "") + prompt


def build_official_reference_prompt(
    task: dict[str, Any],
    screenshot: str,
    xml_path: str,
    previous_actions: list[str],
) -> str:
    inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
    references = inputs.get("same_user_action_references")
    references = references if isinstance(references, list) else []
    reference_actions = references[0].get("actions", []) if references and isinstance(references[0], dict) else []
    from .actions import official_action_text

    prompt = f"""You are an Android GUI agent. You are given an instruction and current screenshot and some supplementary information. You need to perform the next action to complete the instruction.

## Input
User_instruction: {inputs.get('instruction', '')}
User_profile: {profile_text(inputs.get('user_profile', {}) if isinstance(inputs.get('user_profile'), dict) else {})}
Screen_width_height: {image_size(screenshot)}
Screen_description: {screen_description(xml_path)}
Actions_reference: {official_action_text([str(value) for value in reference_actions], predicted=False)}
Previous_actions: {official_action_text(previous_actions, predicted=True)}

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
- 'content' should represent the original text at the click point or the description of the icon.
- 'text' should represent all the original text that the user intends to input.
- 'finished()' means that the instruction is completed.
- Screen_description contains content and coordinates of UI elements.
- Actions_reference is a similar past action sequence from the same user.
- Previous_actions contains actions already performed for the current instruction.
- Only one action in Action Space can be taken. Do not output anything other than the action to take.

The action to take:
"""
    return ("<image>" if screenshot else "") + prompt
