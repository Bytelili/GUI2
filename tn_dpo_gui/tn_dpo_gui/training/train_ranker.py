from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tn_dpo_gui.data.collator import RankerBatchCollator
from tn_dpo_gui.encoders.text_encoder import SimpleTextEncoder
from tn_dpo_gui.evaluation.metrics import pair_accuracy_from_scores
from tn_dpo_gui.models.action_ranker import ActionRanker
from tn_dpo_gui.pair_builder.pair_dataset import TNDPOPairDataset, load_pairs
from tn_dpo_gui.training.losses import pairwise_policy_logratio, weighted_dpo_loss
from tn_dpo_gui.utils.io import write_json
from tn_dpo_gui.utils.logging import get_logger
from tn_dpo_gui.utils.seed import set_seed

from .trainer_utils import AverageMeter, SimpleAdam, save_checkpoint, select_device


def train_ranker(config: dict) -> dict:
    logger = get_logger("tn_dpo_gui.train_ranker")
    set_seed(int(config.get("seed", 7)))
    pair_path = Path(config["data"]["pairs_path"])
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = SimpleTextEncoder(**config.get("encoder", {}))
    dataset = TNDPOPairDataset(load_pairs(pair_path), text_encoder=encoder)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 16)),
        shuffle=True,
        collate_fn=RankerBatchCollator(),
    )
    device = select_device(config["training"].get("device"))
    hidden_dim = int(config["model"].get("hidden_dim", 128))
    dropout = float(config["model"].get("dropout", 0.1))
    model = ActionRanker(encoder.output_dim, encoder.output_dim, encoder.output_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = SimpleAdam(model.parameters(), lr=float(config["training"].get("lr", 1e-3)))
    beta = float(config["training"].get("beta", 1.0))
    epochs = int(config["training"].get("epochs", 3))

    for epoch in range(epochs):
        meter = AverageMeter()
        model.train()
        for batch in loader:
            state_vec = batch["state_vec"].to(device)
            user_vec = batch["user_vec"].to(device)
            chosen_vec = batch["chosen_vec"].to(device)
            rejected_vec = batch["rejected_vec"].to(device)
            chosen_logp_ref = batch["chosen_logp_ref"].to(device)
            rejected_logp_ref = batch["rejected_logp_ref"].to(device)
            weights = batch["weight"].to(device)

            optimizer.zero_grad()
            chosen_scores = model(state_vec, user_vec, chosen_vec)
            rejected_scores = model(state_vec, user_vec, rejected_vec)
            policy_logratio = pairwise_policy_logratio(chosen_scores, rejected_scores)
            ref_logratio = chosen_logp_ref - rejected_logp_ref
            loss = weighted_dpo_loss(policy_logratio, ref_logratio, beta=beta, weights=weights)
            loss.backward()
            optimizer.step()
            meter.update(float(loss.item()), n=len(state_vec))
        logger.info("epoch=%s loss=%.4f", epoch + 1, meter.average)

    model.eval()
    all_chosen_scores = []
    all_rejected_scores = []
    with torch.no_grad():
        for batch in loader:
            state_vec = batch["state_vec"].to(device)
            user_vec = batch["user_vec"].to(device)
            chosen_vec = batch["chosen_vec"].to(device)
            rejected_vec = batch["rejected_vec"].to(device)
            all_chosen_scores.extend(model(state_vec, user_vec, chosen_vec).cpu().tolist())
            all_rejected_scores.extend(model(state_vec, user_vec, rejected_vec).cpu().tolist())

    metrics = {
        "train_loss": meter.average,
        "pair_accuracy": pair_accuracy_from_scores(all_chosen_scores, all_rejected_scores),
        "num_pairs": len(dataset),
    }
    checkpoint = {
        "model_state": model.state_dict(),
        "encoder_config": encoder.get_config(),
        "model_config": {"hidden_dim": hidden_dim, "dropout": dropout},
        "base_model_path": str(config["training"].get("base_model_path", "")),
        "metrics": metrics,
    }
    save_checkpoint(output_dir / "ranker.pt", checkpoint)
    write_json(output_dir / "ranker_metrics.json", metrics)
    return metrics
