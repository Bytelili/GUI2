from __future__ import annotations

from pathlib import Path

import torch


def select_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu")


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class SimpleAdam:
    """Small Adam implementation to avoid environment-specific torch optimizer import issues."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = [param for param in params if param.requires_grad]
        self.lr = float(lr)
        self.beta1, self.beta2 = betas
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.moments_1 = [torch.zeros_like(param) for param in self.params]
        self.moments_2 = [torch.zeros_like(param) for param in self.params]
        self.step_index = 0

    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self) -> None:
        self.step_index += 1
        bias_correction_1 = 1.0 - self.beta1**self.step_index
        bias_correction_2 = 1.0 - self.beta2**self.step_index
        for param, moment_1, moment_2 in zip(self.params, self.moments_1, self.moments_2):
            if param.grad is None:
                continue
            grad = param.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * param.data
            moment_1.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            moment_2.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            corrected_1 = moment_1 / bias_correction_1
            corrected_2 = moment_2 / bias_correction_2
            denominator = corrected_2.sqrt().add_(self.eps)
            param.data.addcdiv_(corrected_1, denominator, value=-self.lr)
