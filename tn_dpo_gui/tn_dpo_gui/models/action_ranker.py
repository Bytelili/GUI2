from __future__ import annotations

import torch
from torch import nn


class ActionRanker(nn.Module):
    def __init__(self, state_dim: int, user_dim: int, action_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.user_proj = nn.Linear(user_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_vec: torch.Tensor, user_vec: torch.Tensor, action_vec: torch.Tensor) -> torch.Tensor:
        state_hidden = torch.tanh(self.state_proj(state_vec))
        user_hidden = torch.tanh(self.user_proj(user_vec))
        action_hidden = torch.tanh(self.action_proj(action_vec))
        features = torch.cat(
            [
                state_hidden,
                user_hidden,
                action_hidden,
                state_hidden * action_hidden,
                user_hidden * action_hidden,
                state_hidden - user_hidden,
                torch.abs(state_hidden - action_hidden),
            ],
            dim=-1,
        )
        return self.mlp(features).squeeze(-1)
