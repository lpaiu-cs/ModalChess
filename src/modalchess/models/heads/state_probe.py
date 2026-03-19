"""상태 충실도 probe 헤드."""

from __future__ import annotations

import torch
from torch import nn

from modalchess.utils.square_utils import square_to_coords


class StateProbeHead(nn.Module):
    """인코더 잠재표현에서 현재 보드 상태 타깃을 복원한다."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.piece_proj = nn.Linear(d_model, 12)
        self.side_to_move_proj = nn.Linear(d_model, 1)
        self.castling_proj = nn.Linear(d_model, 4)
        self.en_passant_proj = nn.Linear(d_model, 65)
        self.in_check_proj = nn.Linear(d_model, 1)
        flat_indices = []
        for square in range(64):
            row, col = square_to_coords(square)
            flat_indices.append(row * 8 + col)
        self.register_buffer("flat_indices", torch.tensor(flat_indices, dtype=torch.long), persistent=False)

    def forward(self, tokens: torch.Tensor, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        """상태 복원 logits를 반환한다."""
        batch_size = tokens.size(0)
        piece_square_logits = self.piece_proj(tokens).transpose(1, 2)
        piece_flat = torch.zeros(
            batch_size,
            12,
            64,
            dtype=piece_square_logits.dtype,
            device=piece_square_logits.device,
        )
        piece_flat.index_copy_(2, self.flat_indices, piece_square_logits)
        piece_logits = piece_flat.view(batch_size, 12, 8, 8)
        return {
            "piece_logits": piece_logits,
            "side_to_move_logits": self.side_to_move_proj(pooled).squeeze(-1),
            "castling_logits": self.castling_proj(pooled),
            "en_passant_logits": self.en_passant_proj(pooled),
            "in_check_logits": self.in_check_proj(pooled).squeeze(-1),
        }
