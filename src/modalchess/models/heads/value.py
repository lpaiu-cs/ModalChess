"""스칼라 value 헤드."""

from __future__ import annotations

import torch
from torch import nn


class ValueHead(nn.Module):
    """pooled 보드 표현으로부터 스칼라 value를 예측한다."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        """`[B]` 형태의 스칼라 value logits를 반환한다."""
        return {"value_logits": self.proj(pooled).squeeze(-1)}
