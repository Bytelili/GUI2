from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PapoCandidate:
    action: str
    source: str
    support: int = 1
    state_distance: float = 0.0
    intent_distance: float = 0.0
    valid: bool = True
    example_step_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PapoNode:
    node_id: str
    step_id: str
    user_id: str
    episode_id: str
    step_index: int
    intent: str
    app: str
    state_key: str
    history_ids: list[str] = field(default_factory=list)
    candidates: list[PapoCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidates"] = [c.to_dict() for c in self.candidates]
        return data


@dataclass
class PapoTree:
    tree_id: str
    root_step_id: str
    target_action: str
    nodes: list[PapoNode]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "root_step_id": self.root_step_id,
            "target_action": self.target_action,
            "nodes": [n.to_dict() for n in self.nodes],
            "metadata": self.metadata,
        }

