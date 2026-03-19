"""ModalChess용 공간 보드 인코더."""

from __future__ import annotations

import math

import torch
from torch import nn

from modalchess.models.relation_bias import RelationBias
from modalchess.models.spatial_positional_encoding import SpatialPositionalEncoding2D
from modalchess.utils.square_utils import square_to_coords


class SquarePatchEmbed(nn.Module):
    """square별 히스토리 특징을 토큰 임베딩으로 사상한다."""

    def __init__(self, history_length: int, input_channels: int, d_model: int) -> None:
        super().__init__()
        self.history_length = history_length
        self.input_channels = input_channels
        self.proj = nn.Linear(history_length * input_channels, d_model)
        flat_indices = []
        for square in range(64):
            row, col = square_to_coords(square)
            flat_indices.append(row * 8 + col)
        self.register_buffer("flat_indices", torch.tensor(flat_indices, dtype=torch.long), persistent=False)

    def forward(self, board_planes: torch.Tensor) -> torch.Tensor:
        """`[B, H, C, 8, 8]`을 square 인덱스 순서의 `[B, 64, D]`로 임베딩한다."""
        batch_size, history_length, channels, _, _ = board_planes.shape
        if history_length != self.history_length:
            raise ValueError(
                f"expected history length {self.history_length}, got {history_length}"
            )
        if channels != self.input_channels:
            raise ValueError(
                f"expected input channels {self.input_channels}, got {channels}"
            )
        flattened = board_planes.reshape(batch_size, history_length, channels, 64)
        flattened = flattened.index_select(-1, self.flat_indices)
        square_features = flattened.permute(0, 3, 1, 2).reshape(
            batch_size,
            64,
            history_length * channels,
        )
        return self.proj(square_features)


class MultiHeadSelfAttention(nn.Module):
    """선택형 relation bias를 지원하는 Transformer 스타일 self-attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        relation_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`[B, 64, D]` 토큰에 multi-head attention을 적용한다."""
        batch_size, num_tokens, _ = tokens.shape
        qkv = self.qkv(tokens)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if relation_bias is not None:
            scores = scores + relation_bias.unsqueeze(0)
        attention = scores.softmax(dim=-1)
        attention = self.attn_dropout(attention)
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_model)
        return self.out_dropout(self.out_proj(context))


class TransformerBlock(nn.Module):
    """보드 토큰 처리를 위한 최소 Transformer 블록."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, relation_bias: torch.Tensor | None = None) -> torch.Tensor:
        """attention과 MLP 잔차 업데이트를 수행한다."""
        tokens = tokens + self.attn(self.norm1(tokens), relation_bias=relation_bias)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class BoardEncoder(nn.Module):
    """square-aware 잠재 토큰을 생성하는 공간 보드 인코더."""

    def __init__(
        self,
        history_length: int,
        input_channels: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_relation_bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_embed = SquarePatchEmbed(
            history_length=history_length,
            input_channels=input_channels,
            d_model=d_model,
        )
        self.positional_encoding = SpatialPositionalEncoding2D(d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.relation_bias = RelationBias(num_heads=num_heads) if use_relation_bias else None
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, board_planes: torch.Tensor) -> dict[str, torch.Tensor]:
        """보드 plane을 square 토큰과 pooled 표현으로 인코딩한다."""
        tokens = self.patch_embed(board_planes)
        tokens = self.positional_encoding(tokens)
        tokens = self.dropout(tokens)
        relation_bias = self.relation_bias() if self.relation_bias is not None else None
        for block in self.blocks:
            tokens = block(tokens, relation_bias=relation_bias)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return {"tokens": tokens, "pooled": pooled}
