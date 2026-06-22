from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .action_schema import Action


def _coerce_actions(actions: list[Action] | list[dict[str, Any]] | None) -> list[Action]:
    return [Action.from_dict(action) for action in actions or []]


@dataclass(slots=True)
class GUIStepExample:
    example_id: str
    user_id: str
    task_id: str
    instruction: str
    state_id: str
    screenshot_path: str | None = None
    ui_tree: str | None = None
    action_history: list[Action] = field(default_factory=list)
    current_action: Action = field(default_factory=lambda: Action(action_type="unknown"))
    future_trajectory: list[Action] = field(default_factory=list)
    task_success: float = 0.0
    progress: float = 0.0
    goal_state: str | None = None
    invalid_count: int = 0
    risk_score: float = 0.0
    split: str = "train"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GUIStepExample":
        return cls(
            example_id=str(payload["example_id"]),
            user_id=str(payload["user_id"]),
            task_id=str(payload["task_id"]),
            instruction=str(payload["instruction"]),
            state_id=str(payload["state_id"]),
            screenshot_path=payload.get("screenshot_path"),
            ui_tree=payload.get("ui_tree"),
            action_history=_coerce_actions(payload.get("action_history")),
            current_action=Action.from_dict(payload.get("current_action") or {}),
            future_trajectory=_coerce_actions(payload.get("future_trajectory")),
            task_success=float(payload.get("task_success", 0.0)),
            progress=float(payload.get("progress", 0.0)),
            goal_state=payload.get("goal_state"),
            invalid_count=int(payload.get("invalid_count", 0)),
            risk_score=float(payload.get("risk_score", 0.0)),
            split=str(payload.get("split", "train")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "state_id": self.state_id,
            "screenshot_path": self.screenshot_path,
            "ui_tree": self.ui_tree,
            "action_history": [action.to_dict() for action in self.action_history],
            "current_action": self.current_action.to_dict(),
            "future_trajectory": [action.to_dict() for action in self.future_trajectory],
            "task_success": self.task_success,
            "progress": self.progress,
            "goal_state": self.goal_state,
            "invalid_count": self.invalid_count,
            "risk_score": self.risk_score,
            "split": self.split,
        }


@dataclass(slots=True)
class TrajectoryRecord:
    trajectory_id: str
    user_id: str
    task_id: str
    instruction: str
    actions: list[Action] = field(default_factory=list)
    states: list[str] | None = None
    task_success: float = 0.0
    progress: float = 0.0
    goal_state: str | None = None
    invalid_count: int = 0
    risk_score: float = 0.0
    split: str = "train"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrajectoryRecord":
        return cls(
            trajectory_id=str(payload["trajectory_id"]),
            user_id=str(payload["user_id"]),
            task_id=str(payload["task_id"]),
            instruction=str(payload["instruction"]),
            actions=_coerce_actions(payload.get("actions")),
            states=list(payload["states"]) if payload.get("states") is not None else None,
            task_success=float(payload.get("task_success", 0.0)),
            progress=float(payload.get("progress", 0.0)),
            goal_state=payload.get("goal_state"),
            invalid_count=int(payload.get("invalid_count", 0)),
            risk_score=float(payload.get("risk_score", 0.0)),
            split=str(payload.get("split", "train")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "actions": [action.to_dict() for action in self.actions],
            "states": self.states,
            "task_success": self.task_success,
            "progress": self.progress,
            "goal_state": self.goal_state,
            "invalid_count": self.invalid_count,
            "risk_score": self.risk_score,
            "split": self.split,
        }


@dataclass(slots=True)
class TrajectoryContinuation:
    source_example_id: str
    source_action: Action
    instruction: str
    actions: list[Action] = field(default_factory=list)
    task_success: float = 0.0
    progress: float = 0.0
    goal_state: str | None = None
    invalid_count: int = 0
    risk_score: float = 0.0
    retrieval_score: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrajectoryContinuation":
        return cls(
            source_example_id=str(payload.get("source_example_id", "")),
            source_action=Action.from_dict(payload.get("source_action") or {}),
            instruction=str(payload.get("instruction", "")),
            actions=_coerce_actions(payload.get("actions")),
            task_success=float(payload.get("task_success", 0.0)),
            progress=float(payload.get("progress", 0.0)),
            goal_state=payload.get("goal_state"),
            invalid_count=int(payload.get("invalid_count", 0)),
            risk_score=float(payload.get("risk_score", 0.0)),
            retrieval_score=float(payload.get("retrieval_score", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_example_id": self.source_example_id,
            "source_action": self.source_action.to_dict(),
            "instruction": self.instruction,
            "actions": [action.to_dict() for action in self.actions],
            "task_success": self.task_success,
            "progress": self.progress,
            "goal_state": self.goal_state,
            "invalid_count": self.invalid_count,
            "risk_score": self.risk_score,
            "retrieval_score": self.retrieval_score,
        }
