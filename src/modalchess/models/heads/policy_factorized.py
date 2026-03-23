"""Factorized move policy 헤드."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class PolicyFactorizedHead(nn.Module):
    """출발칸, 도착칸, 프로모션 logits를 예측한다."""

    def __init__(self, d_model: int, use_pair_scorer: bool = False) -> None:
        super().__init__()
        self.src_proj = nn.Linear(d_model, 1)
        self.dst_proj = nn.Linear(d_model, 1)
        self.promo_proj = nn.Linear(d_model, 5)
        self.use_pair_scorer = use_pair_scorer
        self.pair_scorer = None
        if use_pair_scorer:
            self.pair_scorer = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, tokens: torch.Tensor, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        """factorized move logits를 반환한다."""
        src_logits = self.src_proj(tokens).squeeze(-1)
        dst_logits = self.dst_proj(tokens).squeeze(-1)
        promo_logits = self.promo_proj(pooled)
        outputs: dict[str, torch.Tensor] = {
            "src_logits": src_logits,
            "dst_logits": dst_logits,
            "promo_logits": promo_logits,
        }
        if self.pair_scorer is not None:
            src_tokens = tokens.unsqueeze(2).expand(-1, -1, 64, -1)
            dst_tokens = tokens.unsqueeze(1).expand(-1, 64, -1, -1)
            pair_features = torch.cat([src_tokens, dst_tokens], dim=-1)
            outputs["pair_logits"] = self.pair_scorer(pair_features).squeeze(-1)
        return outputs


def score_factorized_moves(
    policy_outputs: dict[str, torch.Tensor],
    legal_moves: Sequence[tuple[int, int, int]],
) -> torch.Tensor:
    """factorized baseline 공식을 사용해 합법 수의 점수를 계산한다."""
    raw_scores = build_raw_action_scores(policy_outputs)
    scores = []
    for src_square, dst_square, promotion in legal_moves:
        scores.append(raw_scores[src_square, dst_square, promotion])
    if not scores:
        return torch.empty(0, dtype=raw_scores.dtype, device=raw_scores.device)
    return torch.stack(scores)


def build_raw_action_scores(policy_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """raw action-space 전체 [64, 64, 5] 점수 텐서를 구성한다."""
    src_logits = policy_outputs["src_logits"]
    dst_logits = policy_outputs["dst_logits"]
    promo_logits = policy_outputs["promo_logits"]
    pair_logits = policy_outputs.get("pair_logits")
    if src_logits.ndim != 1 or src_logits.size(0) != 64:
        raise ValueError("src_logits는 [64] shape이어야 한다.")
    if dst_logits.ndim != 1 or dst_logits.size(0) != 64:
        raise ValueError("dst_logits는 [64] shape이어야 한다.")
    if promo_logits.ndim != 1 or promo_logits.size(0) != 5:
        raise ValueError("promo_logits는 [5] shape이어야 한다.")
    scores = (
        src_logits[:, None, None]
        + dst_logits[None, :, None]
        + promo_logits[None, None, :]
    )
    if pair_logits is not None:
        if pair_logits.ndim != 2 or pair_logits.shape != (64, 64):
            raise ValueError("pair_logits는 [64, 64] shape이어야 한다.")
        scores = scores + pair_logits[:, :, None]
    return scores
