from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PairMargins:
    task_margin: float
    preference_margin: float
    null_margin: float
    uncertainty: float
    omega: float
    init_weight: float
    final_weight: float
    projection_rho: float
    task_distance: float
