"""공간 보드 상태 입력을 위한 텐서 코덱."""

from __future__ import annotations

from typing import Final, Sequence

import chess
import torch

from modalchess.data.board_state import board_state_to_board
from modalchess.data.fen_codec import fen_to_board_state
from modalchess.data.move_codec import move_to_factorized
from modalchess.data.schema import BoardState
from modalchess.utils.square_utils import BOARD_SIZE, square_to_coords

PIECE_CHANNELS: Final[tuple[str, ...]] = (
    "P",
    "N",
    "B",
    "R",
    "Q",
    "K",
    "p",
    "n",
    "b",
    "r",
    "q",
    "k",
)
NUM_BASE_CHANNELS: Final[int] = 18
EN_PASSANT_NONE_INDEX: Final[int] = 64
SQUARE_STATE_VOCAB: Final[tuple[str, ...]] = ("empty",) + PIECE_CHANNELS
PIECE_TO_SQUARE_STATE_INDEX: Final[dict[str, int]] = {
    symbol: index for index, symbol in enumerate(SQUARE_STATE_VOCAB) if index > 0
}


def empty_board_planes(num_channels: int = NUM_BASE_CHANNELS) -> torch.Tensor:
    """단일 스냅샷용 빈 보드 텐서를 생성한다."""
    return torch.zeros(num_channels, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)


def encode_board_state(state: BoardState) -> torch.Tensor:
    """단일 `BoardState`를 `[C, 8, 8]` plane으로 인코딩한다."""
    planes = empty_board_planes()
    for channel, symbol in enumerate(PIECE_CHANNELS):
        for square, piece_symbol in state.pieces.items():
            if piece_symbol == symbol:
                row, col = square_to_coords(square)
                planes[channel, row, col] = 1.0

    side_to_move_value = 1.0 if state.meta.side_to_move == "w" else 0.0
    planes[12].fill_(side_to_move_value)
    planes[13].fill_(float(state.meta.white_can_castle_kingside))
    planes[14].fill_(float(state.meta.white_can_castle_queenside))
    planes[15].fill_(float(state.meta.black_can_castle_kingside))
    planes[16].fill_(float(state.meta.black_can_castle_queenside))
    if state.meta.en_passant_square is not None:
        row, col = square_to_coords(state.meta.en_passant_square)
        planes[17, row, col] = 1.0
    return planes


def encode_history(states: Sequence[BoardState], history_length: int) -> torch.Tensor:
    """상태 히스토리를 `[H, C, 8, 8]`로 인코딩하고 왼쪽을 0으로 패딩한다."""
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    encoded = torch.stack([encode_board_state(state) for state in states[-history_length:]], dim=0)
    if encoded.size(0) == history_length:
        return encoded
    padding = torch.zeros(
        history_length - encoded.size(0),
        NUM_BASE_CHANNELS,
        BOARD_SIZE,
        BOARD_SIZE,
        dtype=torch.float32,
    )
    return torch.cat([padding, encoded], dim=0)


def encode_fen_history(history_fens: Sequence[str], history_length: int) -> torch.Tensor:
    """FEN 히스토리를 `[H, C, 8, 8]` 텐서로 인코딩한다."""
    states = [fen_to_board_state(fen) for fen in history_fens]
    return encode_history(states, history_length)


def current_snapshot(board_planes: torch.Tensor) -> torch.Tensor:
    """`[H, C, 8, 8]`에서 현재 시점 `[C, 8, 8]` 스냅샷을 꺼낸다."""
    if board_planes.ndim != 4:
        raise ValueError(f"expected `[H, C, 8, 8]`, got shape {tuple(board_planes.shape)}")
    return board_planes[-1]


def build_state_probe_targets(state: BoardState) -> dict[str, torch.Tensor]:
    """`BoardState`로부터 현재 상태 복원용 타깃을 만든다."""
    board = board_state_to_board(state)
    square_state = torch.zeros(BOARD_SIZE, BOARD_SIZE, dtype=torch.long)
    for square, symbol in state.pieces.items():
        row, col = square_to_coords(square)
        square_state[row, col] = PIECE_TO_SQUARE_STATE_INDEX[symbol]
    castling = torch.tensor(
        [
            float(state.meta.white_can_castle_kingside),
            float(state.meta.white_can_castle_queenside),
            float(state.meta.black_can_castle_kingside),
            float(state.meta.black_can_castle_queenside),
        ],
        dtype=torch.float32,
    )
    en_passant_index = state.meta.en_passant_square
    if en_passant_index is None:
        en_passant_index = EN_PASSANT_NONE_INDEX
    return {
        "square_state": square_state,
        "side_to_move": torch.tensor(float(state.meta.side_to_move == "w"), dtype=torch.float32),
        "castling_rights": castling,
        "en_passant_index": torch.tensor(en_passant_index, dtype=torch.long),
        "in_check": torch.tensor(float(board.is_check()), dtype=torch.float32),
    }


def build_legality_tensor(board: chess.Board) -> torch.Tensor:
    """promotion-aware `[64, 64, 5]` legality 타깃 텐서를 만든다."""
    target = torch.zeros(64, 64, 5, dtype=torch.float32)
    for move in board.legal_moves:
        factorized = move_to_factorized(move)
        target[factorized.src_square, factorized.dst_square, factorized.promotion] = 1.0
    return target
