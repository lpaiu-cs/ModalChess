"""합법성 예측 헤드."""

from __future__ import annotations

import torch
from torch import nn


class LegalityHead(nn.Module):
    """promotion-aware `[64, 64, 5]` legality 텐서를 예측한다."""

    def __init__(self, d_model: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.src_proj = nn.Linear(d_model, hidden_dim)
        self.dst_proj = nn.Linear(d_model, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 5)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """legality logits를 반환한다."""
        src = self.src_proj(tokens).unsqueeze(2)
        dst = self.dst_proj(tokens).unsqueeze(1)
        pair_hidden = torch.tanh(src + dst)
        legality_logits = self.out_proj(pair_hidden)
        return {"legality_logits": legality_logits}
