from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from tn_dpo_gui.data.split import filter_by_split
from tn_dpo_gui.encoders.action_encoder import ActionEncoder
from tn_dpo_gui.encoders.text_encoder import SimpleTextEncoder
from tn_dpo_gui.utils.io import read_jsonl

from .pair_schema import TNDPOPair


def filter_pairs_by_split(pairs: list[TNDPOPair], allowed_splits: set[str] | None = None) -> list[TNDPOPair]:
    if not allowed_splits:
        return list(pairs)
    return filter_by_split(pairs, allowed_splits, split_attr="split")


def load_pairs(path: str | Path, allowed_splits: set[str] | None = None) -> list[TNDPOPair]:
    pairs = [TNDPOPair.from_dict(record) for record in read_jsonl(path)]
    return filter_pairs_by_split(pairs, allowed_splits)


def group_pairs_by_state(pairs: list[TNDPOPair]) -> dict[str, list[TNDPOPair]]:
    grouped: dict[str, list[TNDPOPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.state_id].append(pair)
    return dict(grouped)


def summarize_gate_features(state_pairs: list[TNDPOPair], max_candidates: int = 8) -> torch.Tensor:
    candidate_count = max(pair.candidate_count for pair in state_pairs)
    mean_weight = sum(pair.weight for pair in state_pairs) / len(state_pairs)
    mean_uncertainty = sum(pair.uncertainty for pair in state_pairs) / len(state_pairs)
    mean_abs_null = sum(abs(pair.null_margin) for pair in state_pairs) / len(state_pairs)
    mean_ref_gap = sum(abs(pair.chosen_logp_ref - pair.rejected_logp_ref) for pair in state_pairs) / len(state_pairs)
    return torch.tensor(
        [
            candidate_count / max(max_candidates, 1),
            mean_weight,
            mean_uncertainty,
            mean_abs_null,
            mean_ref_gap,
        ],
        dtype=torch.float32,
    )


def gate_target_value(capacity: float, gate_cost: float, target_mode: str = "net_capacity") -> float:
    if target_mode == "net_capacity":
        return float(capacity) - float(gate_cost)
    return float(capacity)


class TNDPOPairDataset(Dataset):
    def __init__(self, pairs: list[TNDPOPair], text_encoder: SimpleTextEncoder | None = None) -> None:
        self.pairs = pairs
        self.text_encoder = text_encoder or SimpleTextEncoder()
        self.action_encoder = ActionEncoder(self.text_encoder)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        return {
            "state_vec": torch.tensor(self.text_encoder.encode_text(pair.state_text), dtype=torch.float32),
            "user_vec": torch.tensor(self.text_encoder.encode_text(pair.user_context_text or pair.history_text), dtype=torch.float32),
            "chosen_vec": torch.tensor(self.action_encoder.encode(pair.chosen_action), dtype=torch.float32),
            "rejected_vec": torch.tensor(self.action_encoder.encode(pair.rejected_action), dtype=torch.float32),
            "chosen_logp_ref": pair.chosen_logp_ref,
            "rejected_logp_ref": pair.rejected_logp_ref,
            "weight": pair.weight,
        }


class GateStateDataset(Dataset):
    def __init__(
        self,
        pairs: list[TNDPOPair],
        text_encoder: SimpleTextEncoder | None = None,
        gate_cost: float = 0.0,
        target_mode: str = "net_capacity",
        max_candidates: int = 8,
    ) -> None:
        self.text_encoder = text_encoder or SimpleTextEncoder()
        grouped = group_pairs_by_state(pairs)
        self.samples = []
        for state_pairs in grouped.values():
            anchor = state_pairs[0]
            self.samples.append(
                {
                    "state_text": anchor.state_text,
                    "user_context_text": anchor.user_context_text,
                    "scalar_features": summarize_gate_features(state_pairs, max_candidates=max_candidates),
                    "target": gate_target_value(anchor.gate_capacity, gate_cost, target_mode=target_mode),
                    "capacity": anchor.gate_capacity,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "state_vec": torch.tensor(self.text_encoder.encode_text(sample["state_text"]), dtype=torch.float32),
            "user_vec": torch.tensor(self.text_encoder.encode_text(sample["user_context_text"]), dtype=torch.float32),
            "scalar_features": sample["scalar_features"],
            "target": sample["target"],
            "capacity": sample["capacity"],
        }
