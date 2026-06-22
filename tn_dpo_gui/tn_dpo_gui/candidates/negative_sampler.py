from __future__ import annotations

from tn_dpo_gui.data.action_normalizer import deduplicate_actions, normalize_action
from tn_dpo_gui.data.action_schema import Action


DEFAULT_NEGATIVES = [
    Action(action_type="click", target="Back"),
    Action(action_type="click", target="Home"),
    Action(action_type="click", target="Menu"),
    Action(action_type="scroll", text="down"),
    Action(action_type="scroll", text="up"),
    Action(action_type="type", target="Search box", text="help"),
]


def sample_negative_actions(current_action, extra_pool: list[Action] | None = None, max_negatives: int = 4) -> list[Action]:
    current_key = normalize_action(current_action).normalized_key()
    pool = deduplicate_actions([*(extra_pool or []), *DEFAULT_NEGATIVES])
    negatives = [action for action in pool if action.normalized_key() != current_key]
    return negatives[:max_negatives]
