from __future__ import annotations

import argparse
from pathlib import Path

from tn_dpo_gui.data.action_schema import Action
from tn_dpo_gui.data.dataset import load_step_examples, load_trajectory_records, save_step_examples, save_trajectory_records
from tn_dpo_gui.data.schema import GUIStepExample, TrajectoryRecord
from tn_dpo_gui.utils.io import ensure_dir, read_json, read_jsonl

from . import PROJECT_ROOT, resolve_path


def _demo_examples() -> tuple[list[GUIStepExample], list[TrajectoryRecord]]:
    examples = [
        GUIStepExample(
            example_id="ex_train_001",
            user_id="user_a",
            task_id="task_music",
            instruction="Search for Taylor Swift songs",
            state_id="state_music_001",
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
            invalid_count=0,
            risk_score=0.05,
            split="train",
        ),
        GUIStepExample(
            example_id="ex_train_002",
            user_id="user_a",
            task_id="task_music",
            instruction="Search for Taylor Swift songs",
            state_id="state_music_002",
            ui_tree="Search box input\nFavorites button\nHome button\nTaylor Swift result button",
            action_history=[Action(action_type="click", target="Search box")],
            current_action=Action(action_type="type", target="Search box", text="Taylor Swift"),
            future_trajectory=[Action(action_type="click", target="Taylor Swift result")],
            task_success=1.0,
            progress=0.95,
            goal_state="Taylor Swift result page",
            invalid_count=0,
            risk_score=0.05,
            split="train",
        ),
        GUIStepExample(
            example_id="ex_train_003",
            user_id="user_b",
            task_id="task_favorites",
            instruction="Open favorite playlist",
            state_id="state_fav_001",
            ui_tree="Favorites button\nSearch box input\nHome button\nScrollable playlists list",
            action_history=[Action(action_type="click", target="Home")],
            current_action=Action(action_type="click", target="Favorites"),
            future_trajectory=[Action(action_type="click", target="Daily Mix")],
            task_success=1.0,
            progress=0.85,
            goal_state="Daily Mix page",
            invalid_count=0,
            risk_score=0.02,
            split="train",
        ),
    ]

    trajectories = [
        TrajectoryRecord(
            trajectory_id="traj_a_001",
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
            invalid_count=0,
            risk_score=0.05,
            split="train",
        ),
        TrajectoryRecord(
            trajectory_id="traj_a_002",
            user_id="user_a",
            task_id="task_music",
            instruction="Search for Taylor Swift songs",
            actions=[
                Action(action_type="click", target="Favorites"),
                Action(action_type="click", target="Taylor Swift playlist"),
            ],
            task_success=0.8,
            progress=0.7,
            goal_state="Taylor Swift playlist",
            invalid_count=0,
            risk_score=0.10,
            split="history",
        ),
        TrajectoryRecord(
            trajectory_id="traj_b_001",
            user_id="user_b",
            task_id="task_music",
            instruction="Search for Taylor Swift songs",
            actions=[
                Action(action_type="scroll", text="down"),
                Action(action_type="click", target="Taylor Swift result"),
            ],
            task_success=0.4,
            progress=0.4,
            goal_state="visible result row",
            invalid_count=1,
            risk_score=0.15,
            split="train",
        ),
        TrajectoryRecord(
            trajectory_id="traj_c_001",
            user_id="user_c",
            task_id="task_music",
            instruction="Search for Taylor Swift songs",
            actions=[
                Action(action_type="click", target="Home"),
                Action(action_type="click", target="Search box"),
            ],
            task_success=0.2,
            progress=0.2,
            goal_state="home page",
            invalid_count=1,
            risk_score=0.20,
            split="train",
        ),
        TrajectoryRecord(
            trajectory_id="traj_b_002",
            user_id="user_b",
            task_id="task_favorites",
            instruction="Open favorite playlist",
            actions=[
                Action(action_type="click", target="Favorites"),
                Action(action_type="click", target="Daily Mix"),
            ],
            task_success=1.0,
            progress=0.9,
            goal_state="Daily Mix page",
            invalid_count=0,
            risk_score=0.02,
            split="train",
        ),
        TrajectoryRecord(
            trajectory_id="traj_b_003",
            user_id="user_b",
            task_id="task_favorites",
            instruction="Open favorite playlist",
            actions=[
                Action(action_type="click", target="Search box"),
                Action(action_type="type", target="Search box", text="Daily Mix"),
            ],
            task_success=0.3,
            progress=0.3,
            goal_state="search page",
            invalid_count=1,
            risk_score=0.12,
            split="history",
        ),
    ]
    return examples, trajectories


def _load_json_or_jsonl(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        if not isinstance(payload, list):
            raise TypeError(f"Expected a JSON list in {path}")
        return payload
    return read_jsonl(path)


def preprocess(output_dir: Path, raw_steps: Path | None = None, raw_trajectories: Path | None = None, demo: bool = False) -> dict[str, str]:
    ensure_dir(output_dir)
    if demo or raw_steps is None or raw_trajectories is None:
        examples, trajectories = _demo_examples()
    else:
        examples = [GUIStepExample.from_dict(item) for item in _load_json_or_jsonl(raw_steps)]
        trajectories = [TrajectoryRecord.from_dict(item) for item in _load_json_or_jsonl(raw_trajectories)]

    step_path = output_dir / "steps.jsonl"
    trajectory_path = output_dir / "trajectories.jsonl"
    save_step_examples(step_path, examples)
    save_trajectory_records(trajectory_path, trajectories)
    return {"steps_path": str(step_path), "trajectories_path": str(trajectory_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize GUI-step and trajectory data for TN-DPO.")
    parser.add_argument("--output-dir", default="data/demo")
    parser.add_argument("--raw-steps")
    parser.add_argument("--raw-trajectories")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    output = preprocess(
        resolve_path(args.output_dir),
        raw_steps=resolve_path(args.raw_steps) if args.raw_steps else None,
        raw_trajectories=resolve_path(args.raw_trajectories) if args.raw_trajectories else None,
        demo=args.demo or (args.raw_steps is None and args.raw_trajectories is None),
    )
    print(output)


if __name__ == "__main__":
    main()
