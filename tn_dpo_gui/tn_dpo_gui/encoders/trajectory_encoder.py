from __future__ import annotations

from tn_dpo_gui.data.action_normalizer import action_texts, normalize_action
from tn_dpo_gui.data.schema import TrajectoryContinuation, TrajectoryRecord

from .text_encoder import SimpleTextEncoder


class TrajectoryEncoder:
    def __init__(self, text_encoder: SimpleTextEncoder) -> None:
        self.text_encoder = text_encoder

    def trajectory_to_text(self, instruction: str, actions: list, goal_state: str | None = None) -> str:
        action_blob = " -> ".join(action_texts(actions)) or "no_future_actions"
        goal_text = goal_state or "unspecified_goal"
        return f"instruction: {instruction}\ntrajectory: {action_blob}\ngoal: {goal_text}"

    def encode_record(self, record: TrajectoryRecord):
        return self.text_encoder.encode_text(self.trajectory_to_text(record.instruction, record.actions, record.goal_state))

    def encode_continuation(self, continuation: TrajectoryContinuation):
        actions = [continuation.source_action] + list(continuation.actions)
        return self.text_encoder.encode_text(self.trajectory_to_text(continuation.instruction, actions, continuation.goal_state))

    def encode_action_continuation(self, instruction: str, action, continuation_actions: list, goal_state: str | None = None):
        seed_action = normalize_action(action)
        return self.text_encoder.encode_text(self.trajectory_to_text(instruction, [seed_action] + list(continuation_actions), goal_state))
