"""FEN 문자열을 선형 입력으로 사용하는 비교용 baseline."""

from __future__ import annotations

import torch
from torch import nn

from modalchess.models.heads.concept import ConceptHead
from modalchess.models.heads.legality import LegalityHead
from modalchess.models.heads.policy_factorized import PolicyFactorizedHead
from modalchess.models.heads.state_probe import StateProbeHead
from modalchess.models.heads.value import ValueHead


class FenPolicyBaselineModel(nn.Module):
    """FEN 문자 시퀀스를 읽고 동일한 출력 공간으로 사상하는 baseline."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        legality_hidden_dim: int = 64,
        concept_vocab: list[str] | None = None,
        use_pair_scorer: bool = False,
    ) -> None:
        super().__init__()
        concept_vocab = concept_vocab or []
        self.max_length = max_length
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.square_queries = nn.Parameter(torch.randn(64, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.policy_head = PolicyFactorizedHead(d_model=d_model, use_pair_scorer=use_pair_scorer)
        self.state_probe_head = StateProbeHead(d_model=d_model)
        self.legality_head = LegalityHead(d_model=d_model, hidden_dim=legality_hidden_dim)
        self.value_head = ValueHead(d_model=d_model)
        self.concept_head = ConceptHead(d_model=d_model, concept_vocab=concept_vocab)

    def forward(
        self,
        board_planes=None,
        meta_features=None,
        fen_token_ids: torch.Tensor | None = None,
        fen_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """FEN 토큰으로부터 policy/state/value 관련 출력을 계산한다."""
        if fen_token_ids is None or fen_attention_mask is None:
            raise ValueError("FEN baseline에는 fen_token_ids와 fen_attention_mask가 필요하다.")
        batch_size, seq_len = fen_token_ids.shape
        if seq_len > self.max_length:
            raise ValueError(f"입력 길이 {seq_len}가 max_length {self.max_length}를 초과한다.")
        positions = torch.arange(seq_len, device=fen_token_ids.device).unsqueeze(0)
        sequence = self.token_embed(fen_token_ids) + self.pos_embed(positions)
        key_padding_mask = ~fen_attention_mask
        sequence = self.encoder(sequence, src_key_padding_mask=key_padding_mask)
        sequence_pooled = (
            sequence * fen_attention_mask.unsqueeze(-1)
        ).sum(dim=1) / fen_attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        square_queries = self.square_queries.unsqueeze(0).expand(batch_size, -1, -1)
        square_tokens, _ = self.cross_attn(
            query=self.query_norm(square_queries),
            key=sequence,
            value=sequence,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        square_tokens = self.output_norm(square_tokens + square_queries)
        board_pooled = square_tokens.mean(dim=1)
        context_pooled = 0.5 * (sequence_pooled + board_pooled)
        outputs = {
            "tokens": square_tokens,
            "meta_tokens": square_tokens.new_zeros(batch_size, 0, square_tokens.size(-1)),
            "context_tokens": square_tokens,
            "board_pooled": board_pooled,
            "context_pooled": context_pooled,
            "pooled": context_pooled,
        }
        outputs.update(self.policy_head(tokens=square_tokens, pooled=context_pooled))
        outputs.update(self.state_probe_head(tokens=square_tokens, pooled=context_pooled))
        outputs.update(self.legality_head(tokens=square_tokens))
        outputs.update(self.value_head(pooled=context_pooled))
        outputs.update(self.concept_head(pooled=context_pooled))
        return outputs
