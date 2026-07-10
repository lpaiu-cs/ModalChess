"""Phase 2 QA 과제 명세: 후보 집합·템플릿·정규 표기 (docs/phase2_plan.md §4).

생성기(qa_generator)와 독립 검증기(qa_verifier)가 공유하는 것은 이 명세(문자열 표기와
후보 구조)뿐이다 — 정답 계산 로직은 양쪽이 서로 다른 python-chess API 경로로 구현한다.
"""

from __future__ import annotations

import chess

EMPTY = "empty"
YES = "yes"
NO = "no"

COLOR_NAMES = {chess.WHITE: "white", chess.BLACK: "black"}
PIECE_WORDS = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def piece_label(color: bool, piece_type: int) -> str:
    """정규 표기: 'white knight' 등."""
    return f"{COLOR_NAMES[color]} {PIECE_WORDS[piece_type]}"


PIECE_ON_SQUARE_CANDIDATES = [EMPTY] + [
    piece_label(color, piece_type)
    for color in (chess.WHITE, chess.BLACK)
    for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
]

# 개정 1: {0,1,2,3+} → 3-way (3+는 승격 의존이라 실데이터 균형 불가)
COUNT_CANDIDATES = ["0", "1", "2 or more"]
COUNT_PIECE_TYPES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)

SIDE_CANDIDATES = ["white", "black"]
CASTLE_SIDES = ("kingside", "queenside")

KING_SQUARE_NUM_CANDIDATES = 8


def no_such_piece(square_name: str, color_name: str | None = None) -> str:
    """거짓 전제 후보의 정규 표기."""
    if color_name is not None:
        return f"there is no {color_name} piece on {square_name}"
    return f"there is no piece on {square_name}"


# 과제별 템플릿. 마지막 인덱스는 held-out(학습 금지, 평가 전용).
TEMPLATES: dict[str, list[str]] = {
    "piece_on_square": [
        "What is on {square}?",
        "Which piece, if any, occupies {square}?",
        "Identify the piece on square {square}.",
        "Look at square {square} and say what stands there.",
    ],
    "king_square": [
        "On which square is the {color} king?",
        "Where is the {color} king located?",
        "Name the square of the {color} king.",
        "Find the {color} king and give its square.",
    ],
    "side_to_move": [
        "Whose turn is it to move?",
        "Which side is to move?",
        "Who moves next in this position?",
        "Say which color has the move.",
    ],
    "castling_right": [
        "Does {color} still have the right to castle {side}?",
        "Does {color} retain {side} castling rights?",
        "Is the {side} castling right still available to {color}?",
        "Check whether {color} can still claim the {side} castling right.",
    ],
    "piece_count": [
        "How many {color} {piece}s are on the board?",
        "Count the {color} {piece}s in this position.",
        "What is the number of {color} {piece}s on the board?",
        "Give the count of {color} {piece}s present.",
    ],
    "square_attacked": [
        "Is the square {square} attacked by {color}?",
        "Does {color} attack the square {square}?",
        "Is {square} under attack by {color}?",
        "Determine if {color} attacks {square}.",
    ],
    "piece_defended": [
        "Is the {color} piece on {square} defended by another {color} piece?",
        "Does another {color} piece defend the {color} piece on {square}?",
        "Is the {color} piece on {square} protected by its own side?",
        "Check whether the {color} piece on {square} has a defender.",
    ],
    "is_check": [
        "Is the side to move currently in check?",
        "Is the player to move in check right now?",
        "Does the side to move stand in check?",
        "State whether the side to move is in check.",
    ],
    "piece_pinned": [
        "Is the piece on {square} pinned to its own king?",
        "Is the piece on {square} absolutely pinned?",
        "Does a pin against its king hold the piece on {square}?",
        "Check if the piece on {square} is pinned to its king.",
    ],
}

HELD_OUT_TEMPLATE_INDEX = 3

TASK_TIER = {
    "piece_on_square": "T1",
    "king_square": "T1",
    "side_to_move": "T1",
    "castling_right": "T1",
    "piece_count": "T1",
    "square_attacked": "T2",
    "piece_defended": "T2",
    "is_check": "T2",
    "piece_pinned": "T2",
}

ALL_TASKS = tuple(TASK_TIER)


def position_key(fen: str) -> str:
    """FEN 앞 4필드(배치/턴/캐슬/ep) — dedup 키."""
    return " ".join(fen.split()[:4])
