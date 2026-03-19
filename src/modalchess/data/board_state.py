"""`python-chess` 위에서 동작하는 BoardState 변환 유틸리티."""

from __future__ import annotations

import chess

from modalchess.data.schema import BoardMeta, BoardState


def board_to_board_state(board: chess.Board) -> BoardState:
    """`python-chess` 보드를 구조화된 `BoardState`로 변환한다."""
    pieces = {
        square: piece.symbol()
        for square, piece in board.piece_map().items()
    }
    meta = BoardMeta(
        side_to_move="w" if board.turn == chess.WHITE else "b",
        white_can_castle_kingside=board.has_kingside_castling_rights(chess.WHITE),
        white_can_castle_queenside=board.has_queenside_castling_rights(chess.WHITE),
        black_can_castle_kingside=board.has_kingside_castling_rights(chess.BLACK),
        black_can_castle_queenside=board.has_queenside_castling_rights(chess.BLACK),
        en_passant_square=board.ep_square,
        halfmove_clock=board.halfmove_clock,
        fullmove_number=board.fullmove_number,
        repetition_count=0,
    )
    return BoardState(pieces=pieces, meta=meta)


def _castling_fen(meta: BoardMeta) -> str:
    rights: list[str] = []
    if meta.white_can_castle_kingside:
        rights.append("K")
    if meta.white_can_castle_queenside:
        rights.append("Q")
    if meta.black_can_castle_kingside:
        rights.append("k")
    if meta.black_can_castle_queenside:
        rights.append("q")
    return "".join(rights) or "-"


def board_state_to_board(state: BoardState) -> chess.Board:
    """`BoardState`로부터 `python-chess` 보드를 다시 만든다."""
    board = chess.Board(None)
    for square, symbol in state.pieces.items():
        board.set_piece_at(square, chess.Piece.from_symbol(symbol))
    board.turn = state.meta.side_to_move == "w"
    board.set_castling_fen(_castling_fen(state.meta))
    board.ep_square = state.meta.en_passant_square
    board.halfmove_clock = state.meta.halfmove_clock
    board.fullmove_number = state.meta.fullmove_number
    return board


def extract_legal_moves_uci(state: BoardState) -> list[str]:
    """주어진 보드 상태의 합법 수를 UCI 표기로 반환한다."""
    board = board_state_to_board(state)
    return [move.uci() for move in board.legal_moves]
