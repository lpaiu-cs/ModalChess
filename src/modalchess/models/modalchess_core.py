"""ModalChess 1단계 코어 모델."""

from __future__ import annotations

from torch import nn

from modalchess.models.board_encoder import BoardEncoder
from modalchess.models.heads.concept import ConceptHead
from modalchess.models.heads.legality import LegalityHead
from modalchess.models.heads.policy_factorized import PolicyFactorizedHead
from modalchess.models.heads.state_probe import StateProbeHead
from modalchess.models.meta_encoder import MetaEncoder
from modalchess.models.heads.value import ValueHead


class ModalChessCoreModel(nn.Module):
    """factorized policy와 보조 헤드를 갖춘 공간 체스 베이스라인."""

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
        legality_hidden_dim: int = 64,
        concept_vocab: list[str] | None = None,
        use_pair_scorer: bool = False,
        meta_num_tokens: int = 2,
        meta_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        concept_vocab = concept_vocab or []
        self.meta_encoder = MetaEncoder(
            d_model=d_model,
            num_tokens=meta_num_tokens,
            hidden_dim=meta_hidden_dim,
        )
        self.encoder = BoardEncoder(
            history_length=history_length,
            input_channels=input_channels,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_relation_bias=use_relation_bias,
        )
        self.policy_head = PolicyFactorizedHead(d_model=d_model, use_pair_scorer=use_pair_scorer)
        self.state_probe_head = StateProbeHead(d_model=d_model)
        self.legality_head = LegalityHead(d_model=d_model, hidden_dim=legality_hidden_dim)
        self.value_head = ValueHead(d_model=d_model)
        self.concept_head = ConceptHead(d_model=d_model, concept_vocab=concept_vocab)

    def forward(
        self,
        board_planes=None,
        meta_features=None,
        fen_token_ids=None,
        fen_attention_mask=None,
    ):
        """`[B, H, C, 8, 8]` 입력에 대해 공간 베이스라인을 실행한다."""
        if board_planes is None:
            raise ValueError("spatial baseline에는 board_planes가 필요하다.")
        extra_tokens = None
        if meta_features is not None:
            extra_tokens = self.meta_encoder(meta_features)
        encoded = self.encoder(board_planes, extra_tokens=extra_tokens)
        tokens = encoded["tokens"]
        pooled = encoded["pooled"]
        outputs = {}
        outputs.update(encoded)
        outputs.update(self.policy_head(tokens=tokens, pooled=pooled))
        outputs.update(self.state_probe_head(tokens=tokens, pooled=pooled))
        outputs.update(self.legality_head(tokens=tokens))
        outputs.update(self.value_head(pooled=pooled))
        outputs.update(self.concept_head(pooled=pooled))
        return outputs
