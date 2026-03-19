from modalchess.utils.square_utils import coords_to_square, square_to_coords


def test_square_coordinate_round_trip() -> None:
    for square in range(64):
        row, col = square_to_coords(square)
        assert coords_to_square(row, col) == square


def test_explicit_coordinate_convention() -> None:
    assert square_to_coords(0) == (7, 0)  # a1 좌표
    assert square_to_coords(7) == (7, 7)  # h1 좌표
    assert square_to_coords(56) == (0, 0)  # a8 좌표
    assert square_to_coords(63) == (0, 7)  # h8 좌표
