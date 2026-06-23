from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tn_dpo_gui.data.collator import RankerBatchCollator
from tn_dpo_gui.encoders.text_encoder import SimpleTextEncoder
from tn_dpo_gui.models.action_ranker import ActionRanker
from tn_dpo_gui.pair_builder.pair_dataset import TNDPOPairDataset, load_pairs

from .metrics import pair_accuracy_from_scores, preference_proxy, projection_metrics, safety_metric, weighted_accuracy_from_scores


def evaluate_ranker(
    pair_path: str | Path,
    checkpoint_path: str | Path,
    batch_size: int = 32,
    allowed_splits: set[str] | None = None,
) -> dict[str, float]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    encoder = SimpleTextEncoder(**checkpoint["encoder_config"])
    pairs = load_pairs(pair_path, allowed_splits=allowed_splits or {"eval"})
    if not pairs:
        raise ValueError(f"No TN-DPO pairs matched evaluation splits {sorted(allowed_splits or {'eval'})} in {pair_path}")
    dataset = TNDPOPairDataset(pairs, text_encoder=encoder)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=RankerBatchCollator())
    model = ActionRanker(
        encoder.output_dim,
        encoder.output_dim,
        encoder.output_dim,
        hidden_dim=int(checkpoint["model_config"]["hidden_dim"]),
        dropout=float(checkpoint["model_config"]["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    chosen_scores = []
    rejected_scores = []
    with torch.no_grad():
        for batch in loader:
            chosen_scores.extend(model(batch["state_vec"], batch["user_vec"], batch["chosen_vec"]).tolist())
            rejected_scores.extend(model(batch["state_vec"], batch["user_vec"], batch["rejected_vec"]).tolist())

    metrics = {
        "pair_accuracy": pair_accuracy_from_scores(chosen_scores, rejected_scores),
        "weighted_pair_accuracy": weighted_accuracy_from_scores(chosen_scores, rejected_scores, [pair.weight for pair in pairs]),
        "preference_proxy": preference_proxy(pairs),
        "safety": safety_metric(pairs),
    }
    metrics.update(projection_metrics(pairs))
    return metrics
