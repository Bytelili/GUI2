from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .official_data import complete_raw_index, index_raw_episodes, read_csv_rows


def audit_dataset(raw_root: str | Path, official_root: str | Path | None = None) -> dict[str, Any]:
    raw_root = Path(raw_root)
    indexed = index_raw_episodes(raw_root)
    complete = complete_raw_index(raw_root, indexed)
    action_dirs = set(indexed.values())
    survey_dirs = {path.parent for path in raw_root.rglob("survey_result.json")}
    users = Counter(user_id for user_id, _time in complete)

    result: dict[str, Any] = {
        "raw_root": str(raw_root),
        "indexed_episodes": len(indexed),
        "complete_episodes": len(complete),
        "incomplete_indexed_episodes": len(indexed) - len(complete),
        "survey_without_action": len(survey_dirs - action_dirs),
        "action_without_survey": len(action_dirs - survey_dirs),
        "num_complete_users": len(users),
        "complete_episodes_by_user": users.most_common(),
        "ready_for_full_build": False,
        "coverage": {},
        "warnings": [],
    }

    if official_root is not None:
        official_root = Path(official_root)
        for name in ["train_set.csv", "test_suggestion.csv", "test_execution.csv", "total.csv"]:
            path = official_root / name
            if not path.exists():
                continue
            rows = read_csv_rows(path)
            covered = sum((str(row.get("user_id") or ""), str(row.get("time") or "")) in complete for row in rows)
            result["coverage"][name] = {
                "covered": covered,
                "total": len(rows),
                "rate": covered / max(len(rows), 1),
            }

    total_coverage = result["coverage"].get("total.csv", {})
    result["ready_for_full_build"] = bool(total_coverage and total_coverage.get("covered") == total_coverage.get("total"))
    if result["incomplete_indexed_episodes"]:
        result["warnings"].append("Some indexed episodes are incomplete and will be skipped.")
    if result["survey_without_action"]:
        result["warnings"].append("Some survey files do not yet have action.jsonl files; extraction may still be running.")
    if not result["ready_for_full_build"]:
        result["warnings"].append("Dataset is not fully extracted; use bounded or split-specific builds.")
    return result
