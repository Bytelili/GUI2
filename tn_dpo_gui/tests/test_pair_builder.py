from __future__ import annotations

from tn_dpo_gui.data.action_schema import Action
from tn_dpo_gui.data.schema import GUIStepExample, TrajectoryRecord
from tn_dpo_gui.pair_builder.pair_builder import TNDPOPairBuilder


def test_pair_builder_generates_nullspace_pairs() -> None:
    trajectories = [
        TrajectoryRecord(
            trajectory_id="traj_a",
            user_id="user_a",
            task_id="task_music",
            instruction="Search for Taylor Swift songs",
            actions=[
                Action(action_type="click", target="Search box"),
                Action(action_type="type", target="Search box", text="Taylor Swift"),
                Action(action_type="click", target="Taylor Swift result"),
            ],
            task_success=1.0,
            progress=1.0,
            goal_state="Taylor Swift result page",
            split="train",
        ),
        TrajectoryRecord(
            trajectory_id="traj_b",
            user_id="user_a",
            task_id="task_music",
            instruction="Search for Taylor Swift songs",
            actions=[Action(action_type="click", target="Favorites"), Action(action_type="click", target="Taylor Swift playlist")],
            task_success=0.8,
            progress=0.7,
            goal_state="Taylor Swift playlist",
            split="history",
        ),
        TrajectoryRecord(
            trajectory_id="traj_c",
            user_id="user_b",
            task_id="task_music",
            instruction="Search for Taylor Swift songs",
            actions=[Action(action_type="scroll", text="down"), Action(action_type="click", target="Taylor Swift result")],
            task_success=0.4,
            progress=0.4,
            invalid_count=1,
            risk_score=0.1,
            goal_state="visible result row",
            split="train",
        ),
        TrajectoryRecord(
            trajectory_id="traj_d",
            user_id="user_c",
            task_id="task_music",
            instruction="Search for Taylor Swift songs",
            actions=[Action(action_type="click", target="Home"), Action(action_type="click", target="Search box")],
            task_success=0.2,
            progress=0.2,
            invalid_count=1,
            risk_score=0.2,
            goal_state="home page",
            split="train",
        ),
    ]
    example = GUIStepExample(
        example_id="ex_1",
        user_id="user_a",
        task_id="task_music",
        instruction="Search for Taylor Swift songs",
        state_id="state_1",
        ui_tree="Search box input\nFavorites button\nHome button\nScrollable results list",
        action_history=[Action(action_type="click", target="Home")],
        current_action=Action(action_type="click", target="Search box"),
        future_trajectory=[
            Action(action_type="type", target="Search box", text="Taylor Swift"),
            Action(action_type="click", target="Taylor Swift result"),
        ],
        task_success=1.0,
        progress=0.9,
        goal_state="Taylor Swift result page",
        risk_score=0.05,
        split="train",
    )

    builder = TNDPOPairBuilder(
        trajectories,
        config={"max_candidates": 8, "lambda_u": 0.0, "min_null_margin": 0.0, "allowed_history_splits": ["train", "history"]},
    )
    pairs = builder.build_pairs([example])
    assert pairs
    assert all(pair.example_id == "ex_1" for pair in pairs)
    assert all(pair.gate_capacity >= 0.0 for pair in pairs)
