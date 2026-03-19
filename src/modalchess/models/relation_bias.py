"""보드 attention에 더할 수 있는 선택형 기하 relation bias."""

from __future__ import annotations

import torch
from torch import nn

import chess


def build_relation_index() -> torch.Tensor:
    """모든 square 쌍에 대한 범주형 relation 행렬을 만든다."""
    relation_index = torch.zeros(64, 64, dtype=torch.long)
    for src in range(64):
        src_file = chess.square_file(src)
        src_rank = chess.square_rank(src)
        for dst in range(64):
            if src == dst:
                relation = 0
            else:
                dst_file = chess.square_file(dst)
                dst_rank = chess.square_rank(dst)
                file_delta = abs(src_file - dst_file)
                rank_delta = abs(src_rank - dst_rank)
                if src_rank == dst_rank:
                    relation = 1
                elif src_file == dst_file:
                    relation = 2
                elif file_delta == rank_delta:
                    relation = 3
                elif sorted((file_delta, rank_delta)) == [1, 2]:
                    relation = 4
                elif max(file_delta, rank_delta) == 1:
                    relation = 5
                else:
                    relation = 0
            relation_index[src, dst] = relation
    return relation_index


class RelationBias(nn.Module):
    """단순한 공간 square 관계에 대한 head별 학습 bias."""

    def __init__(self, num_heads: int, num_relations: int = 6) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_relations, num_heads)
        self.register_buffer("relation_index", build_relation_index(), persistent=False)

    def forward(self) -> torch.Tensor:
        """`[num_heads, 64, 64]` 형태의 relation bias를 반환한다."""
        bias = self.embedding(self.relation_index)
        return bias.permute(2, 0, 1)
