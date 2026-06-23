from __future__ import annotations

from tn_dpo_gui.candidates.candidate_generator import CandidateGenerator
from tn_dpo_gui.candidates.negative_sampler import sample_negative_actions
from tn_dpo_gui.data.action_schema import Action
from tn_dpo_gui.data.schema import GUIStepExample


def test_candidate_generator_reuses_current_type_text_for_affordances() -> None:
    example = GUIStepExample(
        example_id="ex_type",
        user_id="user_a",
        task_id="task_music",
        instruction="Search for Taylor Swift songs",
        state_id="state_type",
        ui_tree="Search box input\nFavorites button",
        current_action=Action(action_type="type", target="Search box", text="Taylor Swift"),
        split="train",
    )
    candidates = CandidateGenerator(max_candidates=8).generate(example)
    type_candidates = [action for action in candidates if action.action_type == "type"]
    assert type_candidates
    assert any(action.text == "Taylor Swift" for action in type_candidates)


def test_negative_sampler_uses_preferred_type_text_when_available() -> None:
    negatives = sample_negative_actions(
        Action(action_type="click", target="Home"),
        preferred_type_text="Taylor Swift",
        max_negatives=8,
    )
    assert any(action.action_type == "type" and action.text == "Taylor Swift" for action in negatives)
