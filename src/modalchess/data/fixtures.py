"""스모크 테스트와 베이스라인 실행을 위한 로컬 fixture 포지션."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess


@dataclass(slots=True)
class FixtureSpec:
    """로컬 fixture 포지션을 정의하는 스펙."""

    position_id: str
    setup_moves: list[str] = field(default_factory=list)
    target_move_uci: str | None = None
    concept_tags: list[str] = field(default_factory=list)
    engine_eval_cp: float | None = None
    start_fen: str | None = None


DEFAULT_CONCEPT_VOCAB: tuple[str, ...] = (
    "check",
    "capture",
    "recapture",
    "pin",
    "fork",
    "skewer",
    "discovered_attack",
    "discovered_check",
    "king_safety",
    "passed_pawn",
    "open_file",
    "promotion_threat",
)


FIXTURE_SPECS: tuple[FixtureSpec, ...] = (
    FixtureSpec(
        position_id="start_position",
        target_move_uci="e2e4",
        concept_tags=["king_safety"],
        engine_eval_cp=0.0,
    ),
    FixtureSpec(
        position_id="opening_development",
        setup_moves=["e2e4", "e7e5", "g1f3"],
        target_move_uci="b8c6",
        concept_tags=["king_safety"],
        engine_eval_cp=0.0,
    ),
    FixtureSpec(
        position_id="white_castle",
        setup_moves=["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"],
        target_move_uci="e1g1",
        concept_tags=["king_safety"],
        engine_eval_cp=20.0,
    ),
    FixtureSpec(
        position_id="en_passant",
        setup_moves=["e2e4", "a7a6", "e4e5", "d7d5"],
        target_move_uci="e5d6",
        concept_tags=["capture", "passed_pawn"],
        engine_eval_cp=35.0,
    ),
    FixtureSpec(
        position_id="promotion",
        start_fen="4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        target_move_uci="a7a8q",
        concept_tags=["promotion_threat"],
        engine_eval_cp=900.0,
    ),
    FixtureSpec(
        position_id="tactical_check",
        setup_moves=["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"],
        target_move_uci="h5f7",
        concept_tags=["check", "capture", "fork"],
        engine_eval_cp=500.0,
    ),
)


def build_board_from_spec(spec: FixtureSpec) -> tuple[chess.Board, list[str]]:
    """fixture에 대한 현재 보드와 시간순 FEN 히스토리를 만든다."""
    board = chess.Board(spec.start_fen) if spec.start_fen is not None else chess.Board()
    history = [board.fen(en_passant="fen")]
    for uci in spec.setup_moves:
        board.push(chess.Move.from_uci(uci))
        history.append(board.fen(en_passant="fen"))
    return board, history


def fixture_boards() -> list[tuple[FixtureSpec, chess.Board, list[str]]]:
    """모든 fixture 보드와 FEN 히스토리를 실제 객체로 생성한다."""
    materialized: list[tuple[FixtureSpec, chess.Board, list[str]]] = []
    for spec in FIXTURE_SPECS:
        board, history = build_board_from_spec(spec)
        materialized.append((spec, board, history))
    return materialized
