"""합법성 예측 헤드."""

from __future__ import annotations

import math

import torch
from torch import nn


class LegalityHead(nn.Module):
    """조밀한 `[64, 64]` legality 행렬을 예측한다."""

    def __init__(self, d_model: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.src_proj = nn.Linear(d_model, hidden_dim)
        self.dst_proj = nn.Linear(d_model, hidden_dim)
        self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """legality logits를 반환한다."""
        src = self.src_proj(tokens)
        dst = self.dst_proj(tokens)
        legality_logits = torch.matmul(src, dst.transpose(1, 2)) * self.scale
        return {"legality_logits": legality_logits}
