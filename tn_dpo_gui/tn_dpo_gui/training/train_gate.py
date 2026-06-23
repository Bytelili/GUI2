from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from tn_dpo_gui.data.collator import GateBatchCollator
from tn_dpo_gui.encoders.text_encoder import SimpleTextEncoder
from tn_dpo_gui.evaluation.metrics import regression_metrics
from tn_dpo_gui.models.gate import CapacityGate
from tn_dpo_gui.pair_builder.pair_dataset import GateStateDataset, load_pairs
from tn_dpo_gui.utils.io import write_json
from tn_dpo_gui.utils.logging import get_logger
from tn_dpo_gui.utils.seed import set_seed

from .trainer_utils import AverageMeter, SimpleAdam, save_checkpoint, select_device


def train_gate(config: dict) -> dict:
    logger = get_logger("tn_dpo_gui.train_gate")
    set_seed(int(config.get("seed", 7)))
    pair_path = Path(config["data"]["pairs_path"])
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = SimpleTextEncoder(**config.get("encoder", {}))
    target_mode = str(config["training"].get("target_mode", "net_capacity"))
    gate_cost = float(config["training"].get("gate_cost", 0.1))
    allowed_splits = {str(split).lower() for split in config.get("data", {}).get("splits", ["train"])}
    pairs = load_pairs(pair_path, allowed_splits=allowed_splits)
    if not pairs:
        raise ValueError(f"No TN-DPO pairs matched gate-training splits {sorted(allowed_splits)} in {pair_path}")
    dataset = GateStateDataset(
        pairs,
        text_encoder=encoder,
        gate_cost=gate_cost,
        target_mode=target_mode,
        max_candidates=int(config["training"].get("max_candidates", 8)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 16)),
        shuffle=True,
        collate_fn=GateBatchCollator(),
    )
    device = select_device(config["training"].get("device"))
    hidden_dim = int(config["model"].get("hidden_dim", 64))
    dropout = float(config["model"].get("dropout", 0.1))
    model = CapacityGate(encoder.output_dim, encoder.output_dim, scalar_dim=5, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = SimpleAdam(model.parameters(), lr=float(config["training"].get("lr", 1e-3)))
    loss_fn = nn.HuberLoss()
    epochs = int(config["training"].get("epochs", 3))

    for epoch in range(epochs):
        meter = AverageMeter()
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            preds = model(batch["state_vec"].to(device), batch["user_vec"].to(device), batch["scalar_features"].to(device))
            targets = batch["target"].to(device)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            meter.update(float(loss.item()), n=len(targets))
        logger.info("epoch=%s gate_loss=%.4f", epoch + 1, meter.average)

    model.eval()
    predictions = []
    targets = []
    capacities = []
    gates = []
    with torch.no_grad():
        for batch in loader:
            preds = model(batch["state_vec"].to(device), batch["user_vec"].to(device), batch["scalar_features"].to(device))
            predictions.extend(preds.cpu().tolist())
            targets.extend(batch["target"].tolist())
            capacities.extend(batch["capacity"].tolist())
            gates.extend(
                CapacityGate.gate_value(
                    preds.cpu(),
                    cost=gate_cost,
                    tau_g=float(config["training"].get("tau_g", 0.5)),
                    target_mode=target_mode,
                ).tolist()
            )

    metrics = regression_metrics(predictions, targets)
    metrics.update(
        {
            "avg_gate_value": sum(gates) / max(len(gates), 1),
            "avg_capacity": sum(capacities) / max(len(capacities), 1),
            "splits": sorted(allowed_splits),
        }
    )
    checkpoint = {
        "model_state": model.state_dict(),
        "encoder_config": encoder.get_config(),
        "model_config": {"hidden_dim": hidden_dim, "dropout": dropout},
        "training_config": {
            "gate_cost": gate_cost,
            "tau_g": float(config["training"].get("tau_g", 0.5)),
            "target_mode": target_mode,
            "splits": sorted(allowed_splits),
        },
        "base_model_path": str(config["training"].get("base_model_path", "")),
        "metrics": metrics,
    }
    save_checkpoint(output_dir / "gate.pt", checkpoint)
    write_json(output_dir / "gate_metrics.json", metrics)
    return metrics
