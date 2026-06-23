from __future__ import annotations

from tn_dpo_gui.data.action_normalizer import deduplicate_actions
from tn_dpo_gui.data.schema import GUIStepExample, TrajectoryRecord

from .negative_sampler import sample_negative_actions
from .ui_affordance_parser import extract_affordance_actions


class CandidateGenerator:
    def __init__(self, max_candidates: int = 8) -> None:
        self.max_candidates = max_candidates

    def _preferred_type_text(
        self,
        example: GUIStepExample,
        history_records: list[TrajectoryRecord] | None = None,
    ) -> str | None:
        if example.current_action.action_type == "type" and example.current_action.text:
            return example.current_action.text

        for action in reversed(example.action_history):
            if action.action_type == "type" and action.text:
                return action.text

        for record in reversed(history_records or []):
            for action in reversed(record.actions):
                if action.action_type == "type" and action.text:
                    return action.text
        return None

    def generate(
        self,
        example: GUIStepExample,
        history_records: list[TrajectoryRecord] | None = None,
        base_policy=None,
    ) -> list:
        preferred_type_text = self._preferred_type_text(example, history_records)
        candidates = [example.current_action]
        candidates.extend(example.action_history[-3:])
        for record in history_records or []:
            candidates.extend(record.actions[-2:])
        candidates.extend(
            extract_affordance_actions(
                example.ui_tree,
                max_actions=self.max_candidates,
                preferred_type_text=preferred_type_text,
            )
        )
        candidates.extend(sample_negative_actions(example.current_action, preferred_type_text=preferred_type_text))

        if base_policy is not None and hasattr(base_policy, "suggest_actions"):
            candidates.extend(base_policy.suggest_actions(example))

        deduped = deduplicate_actions(candidates)
        current_key = example.current_action.normalized_key()
        deduped.sort(key=lambda action: 0 if action.normalized_key() == current_key else 1)
        return deduped[: self.max_candidates]
