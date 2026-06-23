from __future__ import annotations

import json

import pytest

from tn_dpo_gui.data.main_project_adapter import convert_main_project_artifacts, parse_action_label


def _write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def test_convert_main_project_artifacts_requires_target_step_coverage(tmp_path) -> None:
    train_task = {
        "input": {"instruction": "Search for Taylor Swift songs", "user_id": "user_a", "time": "2024-01-01T00:00:00"},
        "metadata": {"papo_episode_id": "episode_train_001", "partition": "train"},
        "target": {"actions": ["input:TextField", "submit:Search"]},
    }
    eval_task = {
        "input": {"instruction": "Open favorite playlist", "user_id": "user_b", "time": "2024-01-02T00:00:00"},
        "metadata": {"papo_episode_id": "episode_eval_001", "partition": "eval"},
        "target": {"actions": ["select:Favorites", "submit:Daily Mix"]},
    }
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    steps_path = tmp_path / "steps.jsonl"
    _write_jsonl(train_path, [train_task])
    _write_jsonl(eval_path, [eval_task])
    _write_jsonl(steps_path, [])

    with pytest.raises(ValueError, match="requires PAPO step coverage"):
        convert_main_project_artifacts(train_path, eval_path, steps_path)


def test_parse_action_label_recovers_typed_text_from_raw_action() -> None:
    action = parse_action_label(
        "input:TextField",
        raw_action='type(text="Taylor Swift", coordinates=(100, 200))',
        object_role="TextField",
    )
    assert action.action_type == "type"
    assert action.target == "TextField"
    assert action.text == "Taylor Swift"
