from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clean_text(value: str | None) -> str:
    return (value or "").strip()


@dataclass(slots=True)
class Action:
    action_type: str
    target: str | None = None
    text: str | None = None
    bbox: list[float] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        action_type = self.action_type.lower().strip()
        target = _clean_text(self.target)
        text = _clean_text(self.text)
        if action_type == "click":
            return f"click {target or 'target'}"
        if action_type == "type":
            if target and text:
                return f"type {text} into {target}"
            return f"type {text or 'text'}"
        if action_type == "scroll":
            return f"scroll {text or target or 'down'}"
        if action_type == "select":
            if target and text:
                return f"select {text} from {target}"
            return f"select {text or target or 'item'}"
        if action_type == "hotkey":
            return f"press {text or target or 'shortcut'}"
        details = " ".join(part for part in [target, text] if part)
        return f"{action_type} {details}".strip()

    def normalized_key(self) -> str:
        action_type = self.action_type.lower().strip()
        target = _clean_text(self.target).lower()
        text = _clean_text(self.text).lower()
        return "|".join([action_type, target, text])

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "text": self.text,
            "bbox": self.bbox,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | "Action") -> "Action":
        if isinstance(payload, Action):
            return payload
        return cls(
            action_type=str(payload.get("action_type") or payload.get("type") or "unknown"),
            target=payload.get("target"),
            text=payload.get("text"),
            bbox=list(payload["bbox"]) if payload.get("bbox") is not None else None,
            raw=dict(payload.get("raw") or {}),
        )

    def __str__(self) -> str:
        return self.to_text()
