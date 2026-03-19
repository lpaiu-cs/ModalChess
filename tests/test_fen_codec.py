import chess

from modalchess.data.board_state import board_state_to_board
from modalchess.data.fen_codec import board_state_to_fen, fen_to_board_state
from modalchess.data.fixtures import fixture_boards


def test_fen_round_trip_on_fixtures() -> None:
    for _, board, _ in fixture_boards():
        fen = board.fen(en_passant="fen")
        state = fen_to_board_state(fen)
        assert board_state_to_fen(state) == fen


def test_board_state_round_trip_and_legal_move_agreement() -> None:
    for _, board, _ in fixture_boards():
        state = fen_to_board_state(board.fen(en_passant="fen"))
        rebuilt = board_state_to_board(state)
        assert rebuilt.fen(en_passant="fen") == board.fen(en_passant="fen")
        assert {move.uci() for move in rebuilt.legal_moves} == {move.uci() for move in board.legal_moves}
