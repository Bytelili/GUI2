from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_json


def load_success_rules(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    value = read_json(source)
    rules = value.get("rules")
    return rules if isinstance(rules, dict) else {}


def verify_success(task: dict[str, Any], final_xml_path: str, rules: dict[str, Any]) -> dict[str, Any]:
    identifier = str(task.get("task_id") or "")
    rule = rules.get(identifier)
    if not isinstance(rule, dict):
        return {
            "success": None,
            "success_verified": False,
            "success_source": "",
            "success_evidence": "No explicit final-state rule is configured.",
        }
    predicates = ("xml_contains_all", "xml_contains_any", "xml_not_contains")
    if not any(
        isinstance(rule.get(name), list)
        and any(str(value).strip() for value in rule[name])
        for name in predicates
    ):
        return {
            "success": None,
            "success_verified": False,
            "success_source": str(rule.get("source") or ""),
            "success_evidence": "Success rule has no non-empty final-state predicate.",
        }
    xml_path = Path(final_xml_path)
    if not xml_path.is_file():
        return {
            "success": None,
            "success_verified": False,
            "success_source": str(rule.get("source") or "rule"),
            "success_evidence": "Final XML observation is missing.",
        }
    text = xml_path.read_text(encoding="utf-8", errors="replace")
    contains_all = [str(value) for value in rule.get("xml_contains_all") or []]
    contains_any = [str(value) for value in rule.get("xml_contains_any") or []]
    contains_none = [str(value) for value in rule.get("xml_not_contains") or []]
    failures = []
    if contains_all and not all(value in text for value in contains_all):
        failures.append("xml_contains_all")
    if contains_any and not any(value in text for value in contains_any):
        failures.append("xml_contains_any")
    if contains_none and any(value in text for value in contains_none):
        failures.append("xml_not_contains")
    return {
        "success": not failures,
        "success_verified": True,
        "success_source": str(rule.get("source") or "explicit final-state rule"),
        "success_evidence": "passed" if not failures else f"failed checks: {','.join(failures)}",
    }
