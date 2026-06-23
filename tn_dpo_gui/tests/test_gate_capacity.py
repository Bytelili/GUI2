from __future__ import annotations

import pytest
import torch

from tn_dpo_gui.models.gate import CapacityGate
from tn_dpo_gui.pair_builder.pair_dataset import gate_target_value
from tn_dpo_gui.training.losses import capacity_slack


def test_capacity_slack_matches_formula() -> None:
    assert capacity_slack(0.6, 0.5, lambda_u=0.5) == 0.35


def test_gate_value_is_monotonic() -> None:
    low = CapacityGate.gate_value(torch.tensor([0.1]), cost=0.5, tau_g=0.5)
    high = CapacityGate.gate_value(torch.tensor([1.0]), cost=0.5, tau_g=0.5)
    assert float(high) > float(low)


def test_gate_target_defaults_to_net_capacity() -> None:
    assert gate_target_value(0.8, 0.1) == pytest.approx(0.7)
