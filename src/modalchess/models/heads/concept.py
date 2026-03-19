"""구성 가능한 concept 예측 헤드."""

from __future__ import annotations

import torch
from torch import nn


class ConceptHead(nn.Module):
    """pooled 보드 특징으로부터 멀티라벨 체스 concept를 예측한다."""

    def __init__(self, d_model: int, concept_vocab: list[str]) -> None:
        super().__init__()
        self.concept_vocab = concept_vocab
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(concept_vocab)),
        )

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        """`[B, num_concepts]` 형태의 concept logits를 반환한다."""
        return {"concept_logits": self.proj(pooled)}
