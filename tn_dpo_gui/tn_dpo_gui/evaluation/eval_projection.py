from __future__ import annotations

from pathlib import Path

from tn_dpo_gui.pair_builder.pair_dataset import load_pairs

from .metrics import projection_metrics


def evaluate_projection(pair_path: str | Path, allowed_splits: set[str] | None = None) -> dict[str, float]:
    pairs = load_pairs(pair_path, allowed_splits=allowed_splits or {"eval"})
    if not pairs:
        raise ValueError(f"No TN-DPO pairs matched projection-eval splits {sorted(allowed_splits or {'eval'})} in {pair_path}")
    return projection_metrics(pairs)
