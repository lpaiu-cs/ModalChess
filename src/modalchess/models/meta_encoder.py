"""비-plane 메타데이터를 토큰으로 바꾸는 인코더."""

from __future__ import annotations

import torch
from torch import nn


class MetaEncoder(nn.Module):
    """스칼라 메타데이터를 1개 이상의 meta token으로 변환한다."""

    def __init__(
        self,
        d_model: int,
        num_tokens: int = 2,
        input_dim: int = 3,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * d_model),
        )
        self.register_buffer(
            "feature_scale",
            torch.tensor([100.0, 200.0, 10.0], dtype=torch.float32),
            persistent=False,
        )

    def forward(self, meta_features: torch.Tensor) -> torch.Tensor:
        """`[B, 3]` 메타데이터를 `[B, num_tokens, d_model]`로 변환한다."""
        normalized = meta_features / self.feature_scale.to(meta_features.device)
        tokens = self.proj(normalized)
        return tokens.view(meta_features.size(0), self.num_tokens, -1)
