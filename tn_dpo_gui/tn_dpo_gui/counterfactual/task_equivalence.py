from __future__ import annotations

from tn_dpo_gui.data.schema import TrajectoryContinuation
from tn_dpo_gui.scoring.task_reward import continuation_task_distance


def task_equivalence_distance(left: TrajectoryContinuation, right: TrajectoryContinuation) -> float:
    return continuation_task_distance(left, right)
