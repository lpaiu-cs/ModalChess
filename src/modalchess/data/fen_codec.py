"""FEN과 BoardState 간 변환 코덱."""

from __future__ import annotations

import chess

from modalchess.data.board_state import board_state_to_board, board_to_board_state
from modalchess.data.schema import BoardState


def fen_to_board_state(fen: str) -> BoardState:
    """FEN 문자열을 `BoardState`로 파싱한다."""
    board = chess.Board(fen)
    return board_to_board_state(board)


def board_state_to_fen(state: BoardState) -> str:
    """`BoardState`를 FEN 문자열로 직렬화한다."""
    board = board_state_to_board(state)
    return board.fen(en_passant="fen")
