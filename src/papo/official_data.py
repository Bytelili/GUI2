from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


OFFICIAL_AGE_GROUP_2 = {
    "30", "70", "73", "74", "75", "77", "79", "80",
    "86", "88", "89", "93", "94", "95", "96", "97",
}


@dataclass(frozen=True)
class EpisodeRef:
    user_id: str
    time: str
    scenario: str
    app: str
    intent: str
    intent_class: str

    @property
    def episode_id(self) -> str:
        return f"{self.user_id}__{self.time}"

    def to_dict(self) -> dict[str, str]:
        return {
            "episode_id": self.episode_id,
            "user_id": self.user_id,
            "time": self.time,
            "scenario": self.scenario,
            "app": self.app,
            "intent": self.intent,
            "intent_class": self.intent_class,
        }


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def episode_ref(row: dict[str, Any]) -> EpisodeRef:
    return EpisodeRef(
        user_id=str(row.get("user_id") or ""),
        time=str(row.get("time") or ""),
        scenario=str(row.get("scenario") or ""),
        app=str(row.get("app") or ""),
        intent=str(row.get("intentDescription") or row.get("intent") or ""),
        intent_class=str(row.get("intentClass") or row.get("intent_class") or ""),
    )


def load_profiles(path: str | Path) -> dict[str, dict[str, str]]:
    profiles: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(path):
        user_id = str(row.pop("user_id", "") or "")
        profiles[user_id] = row
    return profiles


def index_catalog(rows: list[dict[str, str]]) -> dict[str, list[EpisodeRef]]:
    by_user: dict[str, list[EpisodeRef]] = {}
    for row in rows:
        ref = episode_ref(row)
        by_user.setdefault(ref.user_id, []).append(ref)
    for episodes in by_user.values():
        episodes.sort(key=lambda item: item.time)
    return by_user


def previous_episodes(
    target: EpisodeRef,
    catalog_by_user: dict[str, list[EpisodeRef]],
    limit: int = 20,
) -> list[EpisodeRef]:
    rows = [row for row in catalog_by_user.get(target.user_id, []) if row.time < target.time]
    return rows[-limit:] if limit > 0 else rows


def intent_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left or "", right or "").ratio()


def official_age_group(user_id: str) -> str:
    return "group_2" if str(user_id) in OFFICIAL_AGE_GROUP_2 else "group_1"


def best_reference(
    target: EpisodeRef,
    candidates: list[EpisodeRef],
    raw_root: str | Path,
    raw_index: dict[tuple[str, str], Path] | None = None,
) -> tuple[EpisodeRef, Path, float] | None:
    ranked: list[tuple[float, EpisodeRef, Path]] = []
    for candidate in candidates:
        episode_dir = (
            raw_index.get((candidate.user_id, candidate.time))
            if raw_index is not None
            else find_episode_dir(raw_root, candidate.user_id, candidate.time)
        )
        if episode_dir is None:
            continue
        ranked.append((intent_similarity(target.intent, candidate.intent), candidate, episode_dir))
    if not ranked:
        return None
    score, ref, episode_dir = max(ranked, key=lambda item: (item[0], item[1].time))
    return ref, episode_dir, score


def best_references(
    target: EpisodeRef,
    candidates: list[EpisodeRef],
    raw_root: str | Path,
    top_k: int,
    raw_index: dict[tuple[str, str], Path] | None = None,
    similarity_threshold: float = 0.0,
    exclude_same_intent: bool = False,
) -> list[tuple[EpisodeRef, Path, float]]:
    ranked: list[tuple[float, EpisodeRef, Path]] = []
    for candidate in candidates:
        if exclude_same_intent and candidate.intent == target.intent:
            continue
        episode_dir = (
            raw_index.get((candidate.user_id, candidate.time))
            if raw_index is not None
            else find_episode_dir(raw_root, candidate.user_id, candidate.time)
        )
        if episode_dir is None:
            continue
        score = intent_similarity(target.intent, candidate.intent)
        if score >= similarity_threshold:
            ranked.append((score, candidate, episode_dir))
    ranked.sort(key=lambda item: (item[0], item[1].time), reverse=True)
    return [(ref, episode_dir, score) for score, ref, episode_dir in ranked[:top_k]]


def find_episode_dir(raw_root: str | Path, user_id: str, time: str) -> Path | None:
    root = Path(raw_root)
    direct_candidates = [root / user_id / time, root / user_id / user_id / time]
    for candidate in direct_candidates:
        if (candidate / "action.jsonl").exists():
            return candidate
    for action_path in root.glob(f"{user_id}/**/{time}/action.jsonl"):
        return action_path.parent
    return None


def index_raw_episodes(raw_root: str | Path) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for action_path in Path(raw_root).rglob("action.jsonl"):
        episode_dir = action_path.parent
        user_candidates = episode_dir.relative_to(raw_root).parts[:-1]
        if not user_candidates:
            continue
        time = episode_dir.name
        for user_id in dict.fromkeys(user_candidates):
            result.setdefault((user_id, time), episode_dir)
    return result


def is_complete_episode(episode_dir: Path) -> bool:
    if not (episode_dir / "action.jsonl").exists() or not (episode_dir / "survey_result.json").exists():
        return False
    for child in episode_dir.iterdir():
        if child.is_dir() and child.name.startswith("Screenshots"):
            has_image = any(path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for path in child.iterdir())
            has_xml = any(path.suffix.lower() == ".xml" for path in child.iterdir())
            return has_image and has_xml
    return False


def complete_raw_index(
    raw_root: str | Path,
    raw_index: dict[tuple[str, str], Path] | None = None,
) -> dict[tuple[str, str], Path]:
    indexed = raw_index if raw_index is not None else index_raw_episodes(raw_root)
    return {key: path for key, path in indexed.items() if is_complete_episode(path)}


def episode_assets(episode_dir: Path | None) -> dict[str, Any]:
    if episode_dir is None:
        return {"episode_path": "", "screenshots": [], "xml_files": [], "actions": []}

    screenshots: list[str] = []
    xml_files: list[str] = []
    for child in episode_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("Screenshots"):
            continue
        screenshots.extend(str(path) for path in child.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        xml_files.extend(str(path) for path in child.iterdir() if path.suffix.lower() == ".xml")

    actions: list[str] = []
    action_path = episode_dir / "action.jsonl"
    if action_path.exists():
        with action_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    actions.append(line)
                    continue
                if isinstance(item, dict):
                    actions.extend(str(value) for value in item.values())
                else:
                    actions.append(str(item))

    return {
        "episode_path": str(episode_dir),
        "screenshots": sorted(screenshots),
        "xml_files": sorted(xml_files),
        "actions": actions,
    }
