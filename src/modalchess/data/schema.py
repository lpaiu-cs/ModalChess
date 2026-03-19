"""ModalChess 포지션과 이동을 위한 구조화된 데이터클래스."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class BoardMeta:
    """체스 포지션의 비-plane 메타데이터를 명시적으로 담는다."""

    side_to_move: str
    white_can_castle_kingside: bool
    white_can_castle_queenside: bool
    black_can_castle_kingside: bool
    black_can_castle_queenside: bool
    en_passant_square: int | None
    halfmove_clock: int
    fullmove_number: int
    repetition_count: int = 0


@dataclass(slots=True)
class BoardState:
    """`python-chess` square를 키로 쓰는 충실한 보드 상태 표현."""

    pieces: dict[int, str]
    meta: BoardMeta


@dataclass(slots=True)
class FactorizedMove:
    """출발칸, 도착칸, 프로모션을 분리한 명시적 이동 표현."""

    src_square: int
    dst_square: int
    promotion: int


@dataclass(slots=True)
class PositionSample:
    """fixture, 학습, 평가에 공통으로 쓰는 단일 포지션 샘플."""

    position_id: str
    game_id: str | None
    fen: str
    history_fens: list[str]
    board_planes: torch.Tensor
    legal_moves_uci: list[str]
    board_state: BoardState
    target_move_uci: str | None = None
    next_fen: str | None = None
    concept_tags: list[str] | None = None
    engine_eval_cp: float | None = None

    @property
    def meta(self) -> BoardMeta:
        """`BoardState`에 포함된 단일 메타데이터 원천을 노출한다."""
        return self.board_state.meta
