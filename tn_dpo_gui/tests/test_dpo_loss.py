from __future__ import annotations

import torch

from tn_dpo_gui.training.losses import weighted_dpo_loss


def test_weighted_dpo_prefers_higher_chosen_margin() -> None:
    weights = torch.tensor([1.0, 1.0])
    better = weighted_dpo_loss(torch.tensor([2.0, 2.0]), torch.tensor([0.0, 0.0]), beta=1.0, weights=weights)
    worse = weighted_dpo_loss(torch.tensor([-1.0, -1.0]), torch.tensor([0.0, 0.0]), beta=1.0, weights=weights)
    assert float(better) < float(worse)
