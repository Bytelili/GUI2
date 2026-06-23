from __future__ import annotations

from tn_dpo_gui.data.action_schema import Action
from tn_dpo_gui.pair_builder.pair_dataset import filter_pairs_by_split
from tn_dpo_gui.pair_builder.pair_schema import TNDPOPair


def _pair(pair_id: str, split: str) -> TNDPOPair:
    action = Action(action_type="click", target="Search box")
    return TNDPOPair(
        pair_id=pair_id,
        example_id=f"ex_{pair_id}",
        user_id="user_a",
        task_id="task_music",
        state_id=f"state_{pair_id}",
        instruction="Search for Taylor Swift songs",
        split=split,
        state_text="state",
        user_context_text="history",
        history_text="history",
        chosen_action=action,
        rejected_action=Action(action_type="click", target="Favorites"),
        chosen_action_text=action.to_text(),
        rejected_action_text="click Favorites",
        chosen_logp_ref=-1.0,
        rejected_logp_ref=-1.0,
        task_margin=0.1,
        preference_margin=0.1,
        null_margin=0.1,
        projection_rho=0.0,
        task_distance=0.1,
        uncertainty=0.1,
        omega=0.9,
        init_weight=1.0,
        weight=0.9,
        gate_capacity=0.4,
        candidate_count=2,
    )


def test_filter_pairs_by_split_keeps_only_requested_splits() -> None:
    pairs = [_pair("train_pair", "train"), _pair("eval_pair", "eval")]
    filtered = filter_pairs_by_split(pairs, {"eval"})
    assert [pair.pair_id for pair in filtered] == ["eval_pair"]
