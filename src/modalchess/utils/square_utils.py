"""코덱과 모델 전반에서 공유하는 square 인덱스 유틸리티."""

from __future__ import annotations

from typing import Final

BOARD_SIZE: Final[int] = 8
NUM_SQUARES: Final[int] = BOARD_SIZE * BOARD_SIZE


def square_to_coords(square: int) -> tuple[int, int]:
    """`python-chess` square 인덱스를 텐서 좌표로 변환한다."""
    if not 0 <= square < NUM_SQUARES:
        raise ValueError(f"square must be in [0, 63], got {square}")
    rank = square // BOARD_SIZE
    file = square % BOARD_SIZE
    row = BOARD_SIZE - 1 - rank
    col = file
    return row, col


def coords_to_square(row: int, col: int) -> int:
    """텐서 좌표를 `python-chess` square 인덱스로 변환한다."""
    if not 0 <= row < BOARD_SIZE or not 0 <= col < BOARD_SIZE:
        raise ValueError(f"row and col must be in [0, 7], got {(row, col)}")
    rank = BOARD_SIZE - 1 - row
    return rank * BOARD_SIZE + col
