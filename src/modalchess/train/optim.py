"""옵티마이저 유틸리티."""

from __future__ import annotations

from torch import nn
from torch.optim import AdamW


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """베이스라인 학습용 기본 옵티마이저를 만든다."""
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
