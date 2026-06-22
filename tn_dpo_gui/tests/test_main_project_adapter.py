from __future__ import annotations

import json

from tn_dpo_gui.data.main_project_adapter import convert_main_project_artifacts


def test_main_project_adapter_preserves_train_eval_and_history_separation(tmp_path) -> None:
    train_tasks = [
        {
            "task_id": "execution__train_ep",
            "input": {
                "user_id": "u1",
                "time": "t1",
                "instruction": "Open settings",
                "same_user_action_references": [
                    {"episode_id": "hist_ep", "user_id": "u1", "time": "t0", "intent": "Open settings", "actions": ["select:Settings"]}
                ],
            },
            "target": {"actions": ["select:Settings"], "intent_class": "settings"},
            "metadata": {"papo_episode_id": "train_ep", "partition": "train"},
        }
    ]
    eval_tasks = [
        {
            "task_id": "execution__eval_ep",
            "input": {
                "user_id": "u2",
                "time": "t2",
                "instruction": "Open settings",
                "same_user_action_references": [
                    {"episode_id": "hist_ep", "user_id": "u1", "time": "t0", "intent": "Open settings", "actions": ["select:Settings"]}
                ],
            },
            "target": {"actions": ["select:Settings"], "intent_class": "settings"},
            "metadata": {"papo_episode_id": "eval_ep", "partition": "eval"},
        }
    ]
    steps = [
        {
            "papo_step_id": "train_ep__0000",
            "episode_id": "train_ep",
            "user_id": "u1",
            "step_index": 0,
            "intent": "Open settings",
            "state_key": "state_a",
            "next_state_key": "state_b",
            "action": "select:Settings",
            "raw_action": "click(settings)",
            "valid_observation": True,
            "action_confidence": 0.9,
            "is_terminal": True,
            "object_tokens": ["Button|Settings"],
        },
        {
            "papo_step_id": "eval_ep__0000",
            "episode_id": "eval_ep",
            "user_id": "u2",
            "step_index": 0,
            "intent": "Open settings",
            "state_key": "state_a",
            "next_state_key": "state_b",
            "action": "select:Settings",
            "raw_action": "click(settings)",
            "valid_observation": True,
            "action_confidence": 0.9,
            "is_terminal": True,
            "object_tokens": ["Button|Settings"],
        },
        {
            "papo_step_id": "hist_ep__0000",
            "episode_id": "hist_ep",
            "user_id": "u1",
            "step_index": 0,
            "intent": "Open settings",
            "state_key": "state_hist",
            "next_state_key": "state_hist_done",
            "action": "select:Settings",
            "raw_action": "click(settings)",
            "valid_observation": True,
            "action_confidence": 0.8,
            "is_terminal": True,
            "object_tokens": ["Button|Settings"],
        },
    ]

    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    steps_path = tmp_path / "steps.jsonl"
    for path, rows in [(train_path, train_tasks), (eval_path, eval_tasks), (steps_path, steps)]:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    examples, trajectories, summary = convert_main_project_artifacts(train_path, eval_path, steps_path)

    assert summary["train_examples"] == 1
    assert summary["eval_examples"] == 1
    assert {example.split for example in examples} == {"train", "eval"}
    assert any(record.split == "train" and record.trajectory_id == "episode::train_ep" for record in trajectories)
    assert any(record.split == "history" and record.trajectory_id == "episode::hist_ep" for record in trajectories)
    assert not any(record.split == "history" and record.trajectory_id == "episode::eval_ep" for record in trajectories)
