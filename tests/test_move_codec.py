import chess

from modalchess.data.board_state import extract_legal_moves_uci
from modalchess.data.fen_codec import fen_to_board_state
from modalchess.data.move_codec import factorized_to_uci, uci_to_factorized


def test_move_factorization_round_trip_including_promotions() -> None:
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    for move in board.legal_moves:
        factorized = uci_to_factorized(move.uci())
        assert factorized_to_uci(factorized) == move.uci()


def test_legal_move_agreement_with_python_chess() -> None:
    fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    state = fen_to_board_state(fen)
    assert set(extract_legal_moves_uci(state)) == {move.uci() for move in chess.Board(fen).legal_moves}
