from __future__ import annotations

from typing import Any


class RankerBatchCollator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        return {
            "state_vec": torch.stack([item["state_vec"] for item in batch]),
            "user_vec": torch.stack([item["user_vec"] for item in batch]),
            "chosen_vec": torch.stack([item["chosen_vec"] for item in batch]),
            "rejected_vec": torch.stack([item["rejected_vec"] for item in batch]),
            "chosen_logp_ref": torch.tensor([item["chosen_logp_ref"] for item in batch], dtype=torch.float32),
            "rejected_logp_ref": torch.tensor([item["rejected_logp_ref"] for item in batch], dtype=torch.float32),
            "weight": torch.tensor([item["weight"] for item in batch], dtype=torch.float32),
        }


class GateBatchCollator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        return {
            "state_vec": torch.stack([item["state_vec"] for item in batch]),
            "user_vec": torch.stack([item["user_vec"] for item in batch]),
            "scalar_features": torch.stack([item["scalar_features"] for item in batch]),
            "target": torch.tensor([item["target"] for item in batch], dtype=torch.float32),
            "capacity": torch.tensor([item["capacity"] for item in batch], dtype=torch.float32),
        }
