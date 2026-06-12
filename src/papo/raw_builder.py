from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class RawAction:
    image_name: str
    timestamp: str
    raw: str
    action_type: str
    x: int | None = None
    y: int | None = None
    text: str = ""
    content: str = ""
    direction: str = ""


@dataclass
class RawUiNode:
    index: int
    text: str
    content_desc: str
    hint: str
    view_id: str
    class_name: str
    bounds: str
    clickable: bool
    enabled: bool
    visible: bool
    scrollable: bool
    editable: bool
    center_x: int | None
    center_y: int | None

    @property
    def merged_text(self) -> str:
        return " ".join(x.strip() for x in [self.text, self.content_desc, self.hint] if x and x.strip())


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()[:length]


def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"raw": line, "_json_error": True})
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def timestamp_from_name(name: str) -> str:
    stem = Path(str(name)).stem
    if stem.startswith("tree_"):
        stem = stem[5:]
    nums = re.findall(r"\d+", stem)
    return nums[-1] if nums else stem


def looks_like_image(name: str) -> bool:
    return Path(str(name).lower()).suffix in IMAGE_SUFFIXES


def parse_bounds(bounds: str) -> tuple[int | None, int | None, int | None, int | None]:
    nums = re.findall(r"-?\d+", bounds or "")
    if len(nums) < 4:
        return None, None, None, None
    return tuple(map(int, nums[:4]))  # type: ignore[return-value]


def center(bounds: str) -> tuple[int | None, int | None]:
    x1, y1, x2, y2 = parse_bounds(bounds)
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None, None
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def point_in_bounds(x: int | None, y: int | None, bounds: str) -> bool:
    if x is None or y is None:
        return False
    x1, y1, x2, y2 = parse_bounds(bounds)
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return False
    return x1 <= x <= x2 and y1 <= y <= y2


def area(bounds: str) -> int:
    x1, y1, x2, y2 = parse_bounds(bounds)
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return 10**12
    return max(0, x2 - x1) * max(0, y2 - y1)


def region(x: int | None, y: int | None, width: int = 1080, height: int = 2400) -> str:
    if x is None or y is None:
        return "unknown"
    vertical = "top" if y < height * 0.25 else "bottom" if y > height * 0.75 else "middle"
    horizontal = "left" if x < width * 0.33 else "right" if x > width * 0.67 else "center"
    return "center" if vertical == "middle" and horizontal == "center" else f"{vertical}_{horizontal}"


def parse_bool(value: str | None) -> bool:
    return str(value or "").lower() == "true"


def parse_xml_nodes(xml_path: Path) -> list[RawUiNode]:
    if not xml_path.exists():
        return []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    nodes: list[RawUiNode] = []
    for idx, elem in enumerate(root.iter("node")):
        bounds = elem.attrib.get("bounds", "")
        cx, cy = center(bounds)
        nodes.append(
            RawUiNode(
                index=idx,
                text=clean_text(elem.attrib.get("text", "")),
                content_desc=clean_text(elem.attrib.get("content_desc", "")),
                hint=clean_text(elem.attrib.get("hint", "")),
                view_id=elem.attrib.get("view_id", "") or elem.attrib.get("resource-id", ""),
                class_name=elem.attrib.get("class", ""),
                bounds=bounds,
                clickable=parse_bool(elem.attrib.get("clickable")),
                enabled=parse_bool(elem.attrib.get("enabled")),
                visible=not (str(elem.attrib.get("visible", "true")).lower() == "false"),
                scrollable=parse_bool(elem.attrib.get("scrollable")),
                editable=parse_bool(elem.attrib.get("editable")),
                center_x=cx,
                center_y=cy,
            )
        )
    return nodes


def parse_action_obj(obj: Any) -> RawAction:
    image_name = ""
    raw = ""
    if isinstance(obj, dict):
        if len(obj) == 1:
            key = next(iter(obj.keys()))
            value = obj[key]
            if looks_like_image(key):
                image_name = str(key)
                raw = str(value)
        if not raw:
            image_name = str(obj.get("image") or obj.get("screenshot") or obj.get("filename") or "")
            raw = str(obj.get("action") or obj.get("operation") or obj.get("raw_action") or obj.get("raw") or "")
    else:
        raw = str(obj)

    action_type = "unknown"
    m = re.match(r"\s*([a-zA-Z_]+)", raw)
    if m:
        action_type = m.group(1)
    if action_type == "press":
        action_type = "press_key"

    x = y = None
    coord = re.search(r"coordinates\s*=\s*\(([-\d]+)\s*,\s*([-\d]+)\)", raw)
    if coord:
        x, y = int(coord.group(1)), int(coord.group(2))

    def quoted(name: str) -> str:
        m1 = re.search(name + r"\s*=\s*'([^']*)'", raw)
        if m1:
            return m1.group(1)
        m2 = re.search(name + r'\s*=\s*"([^"]*)"', raw)
        return m2.group(1) if m2 else ""

    return RawAction(
        image_name=image_name,
        timestamp=timestamp_from_name(image_name) if image_name else "",
        raw=raw,
        action_type=action_type,
        x=x,
        y=y,
        text=quoted("text"),
        content=quoted("content"),
        direction=quoted("direction"),
    )


def find_target_node(nodes: list[RawUiNode], action: RawAction) -> RawUiNode | None:
    candidates = [n for n in nodes if point_in_bounds(action.x, action.y, n.bounds)]
    if not candidates:
        return None
    candidates.sort(key=lambda n: (not n.clickable, area(n.bounds), not n.merged_text, n.index))
    return candidates[0]


def infer_role(node: RawUiNode | None, action: RawAction) -> str:
    if action.action_type == "type":
        return "TextField"
    if node is None:
        return "Unknown"
    cls = node.class_name.lower()
    if node.editable or "edittext" in cls:
        return "TextField"
    if "button" in cls:
        return "Button"
    if node.scrollable:
        return "Scrollable"
    if node.clickable:
        return "Clickable"
    if node.merged_text:
        return "Text"
    return "Unknown"


def semantic_action(action: RawAction, node: RawUiNode | None) -> dict[str, Any]:
    role = infer_role(node, action)
    raw_text = clean_text(action.content or action.text)
    node_text = node.merged_text if node else ""
    text = node_text or raw_text
    canonical = text[:48] if text else role.lower()
    if action.action_type == "scroll":
        label = f"scroll:{action.direction or 'unknown'}"
        semantic = "scroll"
    elif action.action_type == "type":
        label = "input:TextField"
        semantic = "input"
    elif action.action_type in {"back", "press_back"}:
        label = "navigate:back"
        semantic = "navigate"
    elif action.action_type in {"finished", "finish"}:
        label = "finished"
        semantic = "finished"
    elif action.action_type in {"wait"}:
        label = "wait"
        semantic = "wait"
    elif action.action_type in {"click", "long_click"}:
        semantic = "focus" if role == "TextField" else "submit" if role == "Button" else "select"
        label = f"{semantic}:{canonical or 'unknown'}"
    else:
        semantic = action.action_type
        label = action.action_type

    return {
        "action": label,
        "action_type": action.action_type,
        "semantic_verb": semantic,
        "object_role": role,
        "canonical_object": canonical,
        "object_text": text,
        "region": region(action.x, action.y),
        "confidence": 0.9 if node else 0.55 if (action.content or action.text) else 1.0 if action.action_type in {"scroll", "wait", "finished", "finish"} else 0.2,
        "target_index": node.index if node else -1,
        "target_bounds": node.bounds if node else "",
    }


def ui_signature(nodes: list[RawUiNode], max_nodes: int = 40) -> str:
    tokens = []
    for n in nodes[:max_nodes]:
        if not n.visible:
            continue
        text = n.merged_text[:32]
        if text or n.clickable or n.editable or n.scrollable:
            tokens.append("|".join([n.class_name, text, n.view_id[-32:], region(n.center_x, n.center_y)]))
    return "ui:" + stable_hash("||".join(tokens)) if tokens else "ui:empty"


def object_tokens(nodes: list[RawUiNode], max_nodes: int = 48) -> list[str]:
    out: list[str] = []
    for n in nodes[:max_nodes]:
        text = n.merged_text[:32]
        if not (text or n.clickable or n.editable or n.scrollable):
            continue
        role = infer_role(n, RawAction("", "", "", "unknown"))
        out.append("|".join([role, text, n.view_id[-32:], region(n.center_x, n.center_y)]))
    return out


def iter_episode_dirs(
    raw_root: Path,
    require_complete: bool = True,
    selected_episodes: set[tuple[str, str]] | None = None,
    max_episodes_per_user: int = 0,
) -> list[tuple[str, str, Path]]:
    episodes: list[tuple[str, str, Path]] = []
    for action_path in raw_root.rglob("action.jsonl"):
        ep_dir = action_path.parent
        rel = ep_dir.relative_to(raw_root)
        parts = rel.parts
        user_id = parts[0] if parts else ""
        episode_time = ep_dir.name
        if selected_episodes is not None and (user_id, episode_time) not in selected_episodes:
            continue
        if require_complete:
            if not (ep_dir / "survey_result.json").exists():
                continue
            observations = list_observations(ep_dir)
            if not any(item.get("screenshot_path") and item.get("xml_path") for item in observations.values()):
                continue
        episodes.append((user_id, episode_time, ep_dir))
    episodes.sort(key=lambda x: (x[0], x[1], str(x[2])))
    if max_episodes_per_user > 0:
        counts: Counter[str] = Counter()
        limited: list[tuple[str, str, Path]] = []
        for item in episodes:
            if counts[item[0]] >= max_episodes_per_user:
                continue
            limited.append(item)
            counts[item[0]] += 1
        episodes = limited
    return episodes


def list_observations(ep_dir: Path) -> dict[str, dict[str, str]]:
    obs: dict[str, dict[str, str]] = {}
    for child in ep_dir.iterdir():
        if child.is_dir() and child.name.startswith("Screenshots"):
            for path in child.iterdir():
                ts = timestamp_from_name(path.name)
                item = obs.setdefault(ts, {"screenshot_path": "", "xml_path": ""})
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    item["screenshot_path"] = str(path)
                elif path.suffix.lower() == ".xml":
                    item["xml_path"] = str(path)
    return obs


def build_from_raw(
    raw_root: Path,
    max_episodes: int = 0,
    require_complete: bool = True,
    selected_episodes: set[tuple[str, str]] | None = None,
    max_episodes_per_user: int = 0,
    progress_every: int = 25,
    discovered_episode_dirs: list[tuple[str, str, Path]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    episodes_out: list[dict[str, Any]] = []
    steps_out: list[dict[str, Any]] = []
    warnings: list[str] = []
    action_sources = Counter()
    matched_by = Counter()

    print(
        f"raw discovery: scanning {raw_root} for "
        f"{len(selected_episodes) if selected_episodes is not None else 'all'} selected episodes...",
        flush=True,
    )
    episode_dirs = (
        discovered_episode_dirs
        if discovered_episode_dirs is not None
        else iter_episode_dirs(
            raw_root,
            require_complete=require_complete,
            selected_episodes=selected_episodes,
            max_episodes_per_user=max_episodes_per_user,
        )
    )
    if max_episodes > 0:
        episode_dirs = episode_dirs[:max_episodes]
    print(f"raw discovery: found {len(episode_dirs)} complete episodes; parsing XML and actions...", flush=True)

    chronological_rank = 0
    total_episodes = len(episode_dirs)
    for episode_index, (user_id, episode_time, ep_dir) in enumerate(episode_dirs, start=1):
        survey = read_json(ep_dir / "survey_result.json")
        episode_user = str(survey.get("user_id") or user_id)
        app = str(survey.get("app") or "")
        intent = str(survey.get("intentDescription") or survey.get("intent") or "")
        scenario = str(survey.get("scenario") or "")
        time = str(survey.get("time") or episode_time)
        episode_id = f"{episode_user}__{episode_time}"
        observations = list_observations(ep_dir)
        actions = [parse_action_obj(x) for x in read_jsonl(ep_dir / "action.jsonl")]

        episode_steps: list[dict[str, Any]] = []
        for idx, action in enumerate(actions):
            obs = observations.get(action.timestamp, {})
            match = "timestamp" if obs else ""
            if not obs and idx < len(observations):
                ts = sorted(observations.keys())[idx]
                obs = observations[ts]
                match = "index_fallback"
            matched_by[match or "unmatched"] += 1

            xml_path = Path(obs.get("xml_path", ""))
            nodes = parse_xml_nodes(xml_path) if obs.get("xml_path") else []
            target = find_target_node(nodes, action)
            sem = semantic_action(action, target)
            action_sources["xml_point" if target else "raw_action"] += 1
            state = ui_signature(nodes)
            step_id = f"{episode_user}__{episode_time}__{idx:04d}"
            episode_steps.append(
                {
                    "papo_step_id": step_id,
                    "user_id": episode_user,
                    "episode_id": episode_id,
                    "episode_time": episode_time,
                    "episode_path": str(ep_dir),
                    "step_index": idx,
                    "chronological_rank": chronological_rank,
                    "time": time,
                    "app": app,
                    "scenario": scenario,
                    "intent": intent,
                    "intent_key": stable_hash(clean_text(intent), 12),
                    "state_key": f"{app}|{state}",
                    "next_state_key": "",
                    "screenshot_path": obs.get("screenshot_path", ""),
                    "xml_path": obs.get("xml_path", ""),
                    "next_screenshot_path": "",
                    "next_xml_path": "",
                    "action": sem["action"],
                    "action_type": sem["action_type"],
                    "semantic_verb": sem["semantic_verb"],
                    "object_role": sem["object_role"],
                    "canonical_object": sem["canonical_object"],
                    "object_text": sem["object_text"],
                    "region": sem["region"],
                    "action_confidence": sem["confidence"],
                    "target_index": sem["target_index"],
                    "target_bounds": sem["target_bounds"],
                    "valid_observation": bool(obs.get("screenshot_path") and obs.get("xml_path")),
                    "has_next_state": False,
                    "is_terminal": False,
                    "object_tokens": object_tokens(nodes),
                    "raw_action": action.raw,
                    "raw_image_name": action.image_name,
                    "raw_action_timestamp": action.timestamp,
                    "matched_by": match,
                }
            )
            chronological_rank += 1

        for i, step in enumerate(episode_steps):
            next_step = episode_steps[i + 1] if i + 1 < len(episode_steps) else None
            if next_step:
                step["next_state_key"] = next_step["state_key"]
                step["next_screenshot_path"] = next_step["screenshot_path"]
                step["next_xml_path"] = next_step["xml_path"]
                step["has_next_state"] = True
            step["is_terminal"] = next_step is None or step["action"] == "finished"

        episodes_out.append(
            {
                "episode_id": episode_id,
                "user_id": episode_user,
                "time": time,
                "app": app,
                "scenario": scenario,
                "intent": intent,
                "episode_path": str(ep_dir),
                "num_actions": len(actions),
                "num_steps": len(episode_steps),
                "num_observations": len(observations),
            }
        )
        steps_out.extend(episode_steps)
        if progress_every > 0 and (episode_index % progress_every == 0 or episode_index == total_episodes):
            print(
                f"raw progress: {episode_index}/{total_episodes} episodes, {len(steps_out)} steps",
                flush=True,
            )

    users = Counter(s["user_id"] for s in steps_out)
    valid_obs = sum(1 for s in steps_out if s["valid_observation"])
    has_next = sum(1 for s in steps_out if s["has_next_state"])
    confident = sum(1 for s in steps_out if float(s["action_confidence"]) >= 0.5)
    if len(users) < 2:
        warnings.append("Raw sample contains fewer than 2 users; cross-user PAPO cannot be validated locally.")
    audit = {
        "raw_root": str(raw_root),
        "require_complete": require_complete,
        "selection_size": len(selected_episodes) if selected_episodes is not None else None,
        "max_episodes_per_user": max_episodes_per_user,
        "num_episodes": len(episodes_out),
        "num_steps": len(steps_out),
        "num_users": len(users),
        "valid_observation_rate": valid_obs / max(len(steps_out), 1),
        "has_next_state_rate": has_next / max(len(steps_out), 1),
        "confident_action_rate": confident / max(len(steps_out), 1),
        "matched_by": matched_by.most_common(),
        "action_sources": action_sources.most_common(),
        "top_apps": Counter(s["app"] for s in steps_out).most_common(10),
        "top_actions": Counter(s["action"] for s in steps_out).most_common(20),
        "warnings": warnings,
        "papo_raw_contract": {
            "source": "raw/action.jsonl + raw/survey_result.json + raw/Screenshots*/tree_*.xml",
            "uses_previous_processed_steps": False,
            "has_chronological_rank": True,
            "has_transition_state": True,
            "has_action_element_grounding": True,
            "ready_for_offline_counterfactual_tree": True,
        },
    }
    return episodes_out, steps_out, audit
