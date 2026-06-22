from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tn_dpo_gui.data.action_schema import Action


@dataclass(slots=True)
class TNDPOPair:
    pair_id: str
    example_id: str
    user_id: str
    task_id: str
    state_id: str
    instruction: str
    split: str
    state_text: str
    user_context_text: str
    history_text: str
    chosen_action: Action
    rejected_action: Action
    chosen_action_text: str
    rejected_action_text: str
    chosen_logp_ref: float
    rejected_logp_ref: float
    task_margin: float
    preference_margin: float
    null_margin: float
    projection_rho: float
    task_distance: float
    uncertainty: float
    omega: float
    init_weight: float
    weight: float
    gate_capacity: float = 0.0
    candidate_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "example_id": self.example_id,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "state_id": self.state_id,
            "instruction": self.instruction,
            "split": self.split,
            "state_text": self.state_text,
            "user_context_text": self.user_context_text,
            "history_text": self.history_text,
            "chosen_action": self.chosen_action.to_dict(),
            "rejected_action": self.rejected_action.to_dict(),
            "chosen_action_text": self.chosen_action_text,
            "rejected_action_text": self.rejected_action_text,
            "chosen_logp_ref": self.chosen_logp_ref,
            "rejected_logp_ref": self.rejected_logp_ref,
            "task_margin": self.task_margin,
            "preference_margin": self.preference_margin,
            "null_margin": self.null_margin,
            "projection_rho": self.projection_rho,
            "task_distance": self.task_distance,
            "uncertainty": self.uncertainty,
            "omega": self.omega,
            "init_weight": self.init_weight,
            "weight": self.weight,
            "gate_capacity": self.gate_capacity,
            "candidate_count": self.candidate_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TNDPOPair":
        return cls(
            pair_id=str(payload["pair_id"]),
            example_id=str(payload["example_id"]),
            user_id=str(payload["user_id"]),
            task_id=str(payload["task_id"]),
            state_id=str(payload["state_id"]),
            instruction=str(payload["instruction"]),
            split=str(payload["split"]),
            state_text=str(payload["state_text"]),
            user_context_text=str(payload.get("user_context_text", "")),
            history_text=str(payload.get("history_text", "")),
            chosen_action=Action.from_dict(payload["chosen_action"]),
            rejected_action=Action.from_dict(payload["rejected_action"]),
            chosen_action_text=str(payload["chosen_action_text"]),
            rejected_action_text=str(payload["rejected_action_text"]),
            chosen_logp_ref=float(payload["chosen_logp_ref"]),
            rejected_logp_ref=float(payload["rejected_logp_ref"]),
            task_margin=float(payload["task_margin"]),
            preference_margin=float(payload["preference_margin"]),
            null_margin=float(payload["null_margin"]),
            projection_rho=float(payload["projection_rho"]),
            task_distance=float(payload["task_distance"]),
            uncertainty=float(payload["uncertainty"]),
            omega=float(payload["omega"]),
            init_weight=float(payload["init_weight"]),
            weight=float(payload["weight"]),
            gate_capacity=float(payload.get("gate_capacity", 0.0)),
            candidate_count=int(payload.get("candidate_count", 0)),
            metadata=dict(payload.get("metadata") or {}),
        )
