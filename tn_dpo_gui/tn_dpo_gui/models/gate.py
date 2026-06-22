from __future__ import annotations

import torch
from torch import nn


class CapacityGate(nn.Module):
    def __init__(self, state_dim: int, user_dim: int, scalar_dim: int = 5, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.user_proj = nn.Linear(user_dim, hidden_dim)
        self.scalar_proj = nn.Linear(scalar_dim, hidden_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_vec: torch.Tensor, user_vec: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        state_hidden = torch.tanh(self.state_proj(state_vec))
        user_hidden = torch.tanh(self.user_proj(user_vec))
        scalar_hidden = torch.tanh(self.scalar_proj(scalar_features))
        features = torch.cat([state_hidden, user_hidden, state_hidden * user_hidden, scalar_hidden], dim=-1)
        return self.mlp(features).squeeze(-1)

    @staticmethod
    def gate_value(capacity_prediction, cost: float = 0.0, tau_g: float = 0.5, target_mode: str = "capacity"):
        if target_mode == "net_capacity":
            return torch.sigmoid(capacity_prediction / max(float(tau_g), 1e-8))
        return torch.sigmoid((capacity_prediction - float(cost)) / max(float(tau_g), 1e-8))
