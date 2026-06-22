from __future__ import annotations

from tn_dpo_gui.data.action_normalizer import deduplicate_actions
from tn_dpo_gui.data.action_schema import Action


def test_action_text_and_dedup_ignore_bbox() -> None:
    left = Action(action_type="click", target="Favorites", bbox=[0, 0, 10, 10])
    right = Action(action_type="click", target="Favorites", bbox=[1, 1, 11, 11])
    deduped = deduplicate_actions([left, right])
    assert left.to_text() == "click Favorites"
    assert len(deduped) == 1
