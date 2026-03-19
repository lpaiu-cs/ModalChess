"""square-aware 보드 토큰을 위한 2D 위치 인코딩."""

from __future__ import annotations

import torch
from torch import nn

from modalchess.utils.square_utils import square_to_coords


class SpatialPositionalEncoding2D(nn.Module):
    """64개 square 토큰에 대해 학습 가능한 행/열 임베딩을 제공한다."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.row_embed = nn.Embedding(8, d_model)
        self.col_embed = nn.Embedding(8, d_model)
        rows = []
        cols = []
        for square in range(64):
            row, col = square_to_coords(square)
            rows.append(row)
            cols.append(col)
        self.register_buffer("rows", torch.tensor(rows, dtype=torch.long), persistent=False)
        self.register_buffer("cols", torch.tensor(cols, dtype=torch.long), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """`[B, 64, D]` 토큰에 2D 위치 인코딩을 더한다."""
        position = self.row_embed(self.rows) + self.col_embed(self.cols)
        return tokens + position.unsqueeze(0)
