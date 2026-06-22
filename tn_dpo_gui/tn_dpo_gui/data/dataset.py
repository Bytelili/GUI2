from __future__ import annotations

from pathlib import Path

from tn_dpo_gui.utils.io import read_jsonl, write_jsonl

from .schema import GUIStepExample, TrajectoryContinuation, TrajectoryRecord


def load_step_examples(path: str | Path) -> list[GUIStepExample]:
    return [GUIStepExample.from_dict(record) for record in read_jsonl(path)]


def load_trajectory_records(path: str | Path) -> list[TrajectoryRecord]:
    return [TrajectoryRecord.from_dict(record) for record in read_jsonl(path)]


def load_continuations(path: str | Path) -> list[TrajectoryContinuation]:
    return [TrajectoryContinuation.from_dict(record) for record in read_jsonl(path)]


def save_step_examples(path: str | Path, rows: list[GUIStepExample]) -> None:
    write_jsonl(path, [row.to_dict() for row in rows])


def save_trajectory_records(path: str | Path, rows: list[TrajectoryRecord]) -> None:
    write_jsonl(path, [row.to_dict() for row in rows])


def save_continuations(path: str | Path, rows: list[TrajectoryContinuation]) -> None:
    write_jsonl(path, [row.to_dict() for row in rows])
