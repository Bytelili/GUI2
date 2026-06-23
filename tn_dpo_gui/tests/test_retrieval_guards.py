from __future__ import annotations

from tn_dpo_gui.data.action_schema import Action
from tn_dpo_gui.data.schema import GUIStepExample, TrajectoryRecord
from tn_dpo_gui.encoders.text_encoder import SimpleTextEncoder
from tn_dpo_gui.retrieval.continuation_retriever import ContinuationRetriever
from tn_dpo_gui.retrieval.user_history_index import UserHistoryIndex
from tn_dpo_gui.retrieval.user_history_retriever import UserHistoryRetriever


def test_user_history_retriever_excludes_current_trajectory() -> None:
    records = [
        TrajectoryRecord(
            trajectory_id="traj_self",
            user_id="user_a",
            task_id="task_music",
            instruction="Search for Taylor Swift songs",
            actions=[Action(action_type="click", target="Search box")],
            split="train",
        ),
        TrajectoryRecord(
            trajectory_id="traj_history",
            user_id="user_a",
            task_id="task_music",
            instruction="Search for Taylor Swift songs",
            actions=[Action(action_type="click", target="Favorites")],
            split="history",
        ),
    ]
    retriever = UserHistoryRetriever(UserHistoryIndex.build(records), SimpleTextEncoder())
    history = retriever.retrieve(
        "user_a",
        "Search for Taylor Swift songs",
        exclude_trajectory_ids={"traj_self"},
        limit=5,
    )
    assert [record.trajectory_id for record in history] == ["traj_history"]


def test_continuation_retriever_excludes_current_trajectory_records() -> None:
    records = [
        TrajectoryRecord(
            trajectory_id="traj_self",
            user_id="user_a",
            task_id="task_music",
            instruction="Search for Taylor Swift songs",
            actions=[
                Action(action_type="click", target="Search box"),
                Action(action_type="type", target="Search box", text="Taylor Swift"),
            ],
            split="train",
        ),
        TrajectoryRecord(
            trajectory_id="traj_other",
            user_id="user_b",
            task_id="task_music",
            instruction="Search for Taylor Swift songs",
            actions=[
                Action(action_type="click", target="Search box"),
                Action(action_type="click", target="Taylor Swift result"),
            ],
            split="train",
        ),
    ]
    example = GUIStepExample(
        example_id="ex_1",
        user_id="user_a",
        task_id="task_music",
        instruction="Search for Taylor Swift songs",
        state_id="state_1",
        source_trajectory_id="traj_self",
        current_action=Action(action_type="click", target="Search box"),
        future_trajectory=[Action(action_type="type", target="Search box", text="Taylor Swift")],
        split="eval",
    )
    retriever = ContinuationRetriever(records, SimpleTextEncoder(), fallback_current_future=False)
    continuations = retriever.retrieve(example, example.current_action, limit=5)
    assert continuations
    assert all(continuation.source_example_id != "traj_self" for continuation in continuations)
