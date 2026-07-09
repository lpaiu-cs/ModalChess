"""Projection heads + multi-positive symmetric InfoNCE for the alignment connector."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(slots=True)
class ConnectorConfig:
    board_dim: int = 384
    text_dim: int = 384
    proj_dim: int = 128
    hidden_dim: int = 512
    projection: str = "mlp"  # "linear" | "mlp"
    dropout: float = 0.1
    learnable_temperature: bool = True
    init_temperature: float = 0.07
    max_logit_scale: float = 100.0  # clamp exp(logit_scale)


def _build_head(in_dim: int, config: ConnectorConfig) -> nn.Module:
    if config.projection == "linear":
        return nn.Linear(in_dim, config.proj_dim)
    if config.projection == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.proj_dim),
        )
    raise ValueError(f"지원하지 않는 projection: {config.projection}")


class AlignmentConnector(nn.Module):
    """board/text 임베딩을 공유 공간으로 사영하는 두 head + 학습형 온도.

    두 인코더는 frozen이며 여기서 학습하는 것은 projection head와 (선택적) 온도뿐이다.
    """

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__()
        self.config = config
        self.board_head = _build_head(config.board_dim, config)
        self.text_head = _build_head(config.text_dim, config)
        init_logit_scale = math.log(1.0 / config.init_temperature)
        if config.learnable_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale, dtype=torch.float32))
        else:
            self.register_buffer("logit_scale", torch.tensor(init_logit_scale, dtype=torch.float32))

    def encode_board(self, board: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.board_head(board), dim=-1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_head(text), dim=-1)

    def scale(self) -> torch.Tensor:
        return torch.clamp(self.logit_scale.exp(), max=self.config.max_logit_scale)

    def forward(self, board: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_board(board), self.encode_text(text)


def _directional_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    ignore_mask: torch.Tensor,
) -> torch.Tensor:
    """multi-positive InfoNCE 한 방향.

    logits: [N, N] (query=행). positive_mask[i,j]=True면 j는 i의 정답(대각 포함).
    ignore_mask[i,j]=True면 j를 loss에서 제외(positive도 negative도 아님; false negative 방어).
    L_i = logsumexp_j(denominator) - logsumexp_j(positive) 형태.
    """
    neg_inf = torch.finfo(logits.dtype).min
    # 분모: ignore 열을 제외한 모든 열(정답 + 진짜 negative)
    denom_logits = logits.masked_fill(ignore_mask, neg_inf)
    denom = torch.logsumexp(denom_logits, dim=1)
    # 분자: positive 열만
    pos_logits = logits.masked_fill(~positive_mask, neg_inf)
    numer = torch.logsumexp(pos_logits, dim=1)
    return (denom - numer).mean()


def multi_positive_infonce(
    board_proj: torch.Tensor,
    text_proj: torch.Tensor,
    scale: torch.Tensor,
    positive_mask: torch.Tensor,
    ignore_mask: torch.Tensor,
) -> torch.Tensor:
    """symmetric multi-positive InfoNCE.

    board_proj/text_proj: [N, d] (정규화 가정). positive_mask/ignore_mask: [N, N] bool.
    두 mask는 board→text 기준이며, 대칭 방향은 전치해서 쓴다.
    """
    logits = scale * board_proj @ text_proj.transpose(0, 1)
    board_to_text = _directional_loss(logits, positive_mask, ignore_mask)
    text_to_board = _directional_loss(
        logits.transpose(0, 1),
        positive_mask.transpose(0, 1),
        ignore_mask.transpose(0, 1),
    )
    return 0.5 * (board_to_text + text_to_board)


def build_pair_masks(
    position_ids: list[str],
    normalized_texts: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """batch 메타데이터로 positive/ignore mask를 만든다.

    - positive: 같은 position_id(같은 board는 여러 코멘트에 대해 진짜 multi-positive) 또는 대각.
    - ignore: 같은 normalized_text이면서 positive가 아닌 쌍(generic 코멘트의 false negative).
    """
    n = len(position_ids)
    device = "cpu"
    pid = position_ids
    txt = normalized_texts
    pos = torch.eye(n, dtype=torch.bool, device=device)
    same_pid = torch.zeros((n, n), dtype=torch.bool, device=device)
    same_txt = torch.zeros((n, n), dtype=torch.bool, device=device)
    # 소규모 batch 가정(수백) → O(N^2) 파이썬 루프 대신 그룹 인덱스로 채운다.
    from collections import defaultdict

    by_pid: dict[str, list[int]] = defaultdict(list)
    by_txt: dict[str, list[int]] = defaultdict(list)
    for i in range(n):
        by_pid[pid[i]].append(i)
        by_txt[txt[i]].append(i)
    for idxs in by_pid.values():
        if len(idxs) > 1:
            for a in idxs:
                for b in idxs:
                    same_pid[a, b] = True
    for idxs in by_txt.values():
        if len(idxs) > 1:
            for a in idxs:
                for b in idxs:
                    same_txt[a, b] = True
    positive_mask = pos | same_pid
    ignore_mask = same_txt & ~positive_mask
    return positive_mask, ignore_mask
