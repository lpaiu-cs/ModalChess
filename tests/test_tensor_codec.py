import chess
import torch

from modalchess.data.fen_codec import fen_to_board_state
from modalchess.data.move_codec import uci_to_factorized
from modalchess.data.tensor_codec import (
    NUM_BASE_CHANNELS,
    build_legality_tensor,
    build_state_probe_targets,
    current_snapshot,
    encode_board_state,
    encode_fen_history,
)


def test_tensor_codec_invariants_for_start_position() -> None:
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    state = fen_to_board_state(start_fen)
    planes = encode_board_state(state)
    assert planes.shape == (NUM_BASE_CHANNELS, 8, 8)
    assert planes[:12].sum().item() == 32.0
    assert torch.all(planes[12] == 1.0)
    assert torch.all(planes[13:17] == 1.0)
    assert planes[17].sum().item() == 0.0
    assert planes[3, 7, 0].item() == 1.0  # 백 룩이 a1에 있어야 한다.
    assert planes[11, 0, 4].item() == 1.0  # 흑 킹이 e8에 있어야 한다.


def test_history_encoding_and_current_snapshot() -> None:
    history = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    ]
    history_planes = encode_fen_history(history, history_length=3)
    assert history_planes.shape == (3, NUM_BASE_CHANNELS, 8, 8)
    assert history_planes[0].sum().item() == 0.0
    current = current_snapshot(history_planes)
    assert current.shape == (NUM_BASE_CHANNELS, 8, 8)
    assert current[12, 0, 0].item() == 0.0


def test_state_probe_target_shapes() -> None:
    fen = "r3k2r/pppq1ppp/2npbn2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w kq - 0 8"
    state = fen_to_board_state(fen)
    targets = build_state_probe_targets(state)
    assert targets["square_state"].shape == (8, 8)
    assert targets["castling_rights"].shape == (4,)
    assert targets["en_passant_index"].shape == ()


def test_promotion_legality_tensor_is_exact_action_aware() -> None:
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    legality = build_legality_tensor(board)
    assert legality.shape == (64, 64, 5)
    for uci in ("a7a8n", "a7a8b", "a7a8r", "a7a8q"):
        move = uci_to_factorized(uci)
        assert legality[move.src_square, move.dst_square, move.promotion].item() == 1.0
    assert legality[chess.A7, chess.A8, 0].item() == 0.0
