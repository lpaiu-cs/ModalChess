"""UCI 이동과 factorized move 간 변환 코덱."""

from __future__ import annotations

from typing import Final

import chess

from modalchess.data.schema import FactorizedMove

PROMOTION_TO_ID: Final[dict[int | None, int]] = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
ID_TO_PROMOTION: Final[dict[int, int | None]] = {value: key for key, value in PROMOTION_TO_ID.items()}


def move_to_factorized(move: chess.Move) -> FactorizedMove:
    """`python-chess` 이동을 factorized 형식으로 변환한다."""
    return FactorizedMove(
        src_square=move.from_square,
        dst_square=move.to_square,
        promotion=PROMOTION_TO_ID[move.promotion],
    )


def factorized_to_move(move: FactorizedMove) -> chess.Move:
    """factorized move를 `python-chess` 이동으로 변환한다."""
    if move.promotion not in ID_TO_PROMOTION:
        raise ValueError(f"unsupported promotion id: {move.promotion}")
    return chess.Move(
        from_square=move.src_square,
        to_square=move.dst_square,
        promotion=ID_TO_PROMOTION[move.promotion],
    )


def uci_to_factorized(uci: str) -> FactorizedMove:
    """UCI 문자열을 factorized move로 변환한다."""
    return move_to_factorized(chess.Move.from_uci(uci))


def factorized_to_uci(move: FactorizedMove) -> str:
    """factorized move를 UCI 표기로 변환한다."""
    return factorized_to_move(move).uci()
