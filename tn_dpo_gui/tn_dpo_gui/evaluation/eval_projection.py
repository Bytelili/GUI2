from __future__ import annotations

from pathlib import Path

from tn_dpo_gui.pair_builder.pair_dataset import load_pairs

from .metrics import projection_metrics


def evaluate_projection(pair_path: str | Path) -> dict[str, float]:
    return projection_metrics(load_pairs(pair_path))
