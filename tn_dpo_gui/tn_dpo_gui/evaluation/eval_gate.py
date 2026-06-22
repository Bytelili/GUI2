from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tn_dpo_gui.data.collator import GateBatchCollator
from tn_dpo_gui.encoders.text_encoder import SimpleTextEncoder
from tn_dpo_gui.models.gate import CapacityGate
from tn_dpo_gui.pair_builder.pair_dataset import GateStateDataset, load_pairs

from .metrics import regression_metrics


def evaluate_gate(pair_path: str | Path, checkpoint_path: str | Path, batch_size: int = 32) -> dict[str, float]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    encoder = SimpleTextEncoder(**checkpoint["encoder_config"])
    training_config = checkpoint.get("training_config", {})
    dataset = GateStateDataset(
        load_pairs(pair_path),
        text_encoder=encoder,
        gate_cost=float(training_config.get("gate_cost", 0.0)),
        target_mode=str(training_config.get("target_mode", "capacity")),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=GateBatchCollator())
    model = CapacityGate(
        encoder.output_dim,
        encoder.output_dim,
        scalar_dim=5,
        hidden_dim=int(checkpoint["model_config"]["hidden_dim"]),
        dropout=float(checkpoint["model_config"]["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            predictions.extend(model(batch["state_vec"], batch["user_vec"], batch["scalar_features"]).tolist())
            targets.extend(batch["target"].tolist())
    return regression_metrics(predictions, targets)
