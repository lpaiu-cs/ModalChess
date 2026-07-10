"""독립 검증기: 생성기와 다른 API 경로로 정답을 재계산한다 (docs/phase2_plan.md §4).

경로 대비 — generator → verifier:
  piece_at → piece_map / board.king → piece_map 스캔 / board.turn → FEN 필드 파싱 /
  has_*_castling_rights → FEN 캐슬링 필드 파싱 / len(board.pieces()) → piece_map 계수 /
  is_attacked_by → attackers 집합 / board.is_check → board.checkers /
  board.is_pinned → board.pin 마스크.
불일치 1건 = P0 실패.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import chess

from modalchess.fusion.qa_tasks import (
    COUNT_CANDIDATES,
    EMPTY,
    NO,
    PIECE_ON_SQUARE_CANDIDATES,
    PIECE_WORDS,
    SIDE_CANDIDATES,
    TASK_TIER,
    TEMPLATES,
    YES,
    no_such_piece,
    piece_label,
)

_N_CANDIDATES = {
    "piece_on_square": 13,
    "king_square": 8,
    "side_to_move": 2,
    "castling_right": 2,
    "piece_count": 3,
    "square_attacked": 2,
    "piece_defended": 3,
    "is_check": 2,
    "piece_pinned": 3,
}

_WORD_TO_PTYPE = {v: k for k, v in PIECE_WORDS.items()}


def _expected_answer(task: str, fen: str, params: dict[str, Any]) -> str:
    board = chess.Board(fen)
    pieces = board.piece_map()

    if task == "piece_on_square":
        piece = pieces.get(chess.parse_square(params["square"]))
        return EMPTY if piece is None else piece_label(piece.color, piece.piece_type)

    if task == "king_square":
        color = params["color"] == "white"
        for sq, piece in pieces.items():
            if piece.piece_type == chess.KING and piece.color == color:
                return chess.square_name(sq)
        raise ValueError(f"no {params['color']} king in {fen}")

    if task == "side_to_move":
        return "white" if fen.split()[1] == "w" else "black"

    if task == "castling_right":
        castling_field = fen.split()[2]
        char = {"white": {"kingside": "K", "queenside": "Q"},
                "black": {"kingside": "k", "queenside": "q"}}[params["color"]][params["side"]]
        return YES if char in castling_field else NO

    if task == "piece_count":
        color = params["color"] == "white"
        ptype = _WORD_TO_PTYPE[params["piece"]]
        n = sum(1 for p in pieces.values() if p.color == color and p.piece_type == ptype)
        return "2 or more" if n >= 2 else str(n)

    if task == "square_attacked":
        color = params["color"] == "white"
        sq = chess.parse_square(params["square"])
        return YES if len(board.attackers(color, sq)) > 0 else NO

    if task == "piece_defended":
        color = params["color"] == "white"
        sq = chess.parse_square(params["square"])
        piece = pieces.get(sq)
        if piece is None or piece.color != color:
            return no_such_piece(params["square"], params["color"])
        return YES if len(board.attackers(color, sq)) > 0 else NO

    if task == "is_check":
        return YES if bool(board.checkers()) else NO

    if task == "piece_pinned":
        sq = chess.parse_square(params["square"])
        piece = pieces.get(sq)
        if piece is None:
            return no_such_piece(params["square"])
        pinned = board.pin(piece.color, sq) != chess.SquareSet(chess.BB_ALL)
        return YES if pinned else NO

    raise ValueError(f"unknown task {task}")


def verify_item(item: dict[str, Any]) -> list[str]:
    """구조 + 정답 재계산 검증. 빈 리스트 = 통과."""
    errors: list[str] = []
    task = item.get("task")
    if task not in TASK_TIER:
        return [f"unknown task: {task}"]
    if item.get("tier") != TASK_TIER[task]:
        errors.append("tier mismatch")

    candidates = item.get("candidates") or []
    if len(candidates) != _N_CANDIDATES[task]:
        errors.append(f"candidate count {len(candidates)} != {_N_CANDIDATES[task]}")
    if len(set(candidates)) != len(candidates):
        errors.append("duplicate candidates")
    if item.get("answer") not in candidates:
        errors.append("answer not in candidates")

    template_id = item.get("template_id")
    templates = TEMPLATES[task]
    if not isinstance(template_id, int) or not (0 <= template_id < len(templates)):
        errors.append(f"bad template_id {template_id}")
    else:
        expected_q = templates[template_id].format(**item.get("params", {}))
        if item.get("question") != expected_q:
            errors.append("question does not match template+params")
    if "{" in (item.get("question") or ""):
        errors.append("unresolved placeholder in question")

    if task == "piece_on_square" and candidates != PIECE_ON_SQUARE_CANDIDATES:
        errors.append("piece_on_square candidate set mismatch")
    if task == "piece_count" and candidates != COUNT_CANDIDATES:
        errors.append("piece_count candidate set mismatch")
    if task == "side_to_move" and candidates != SIDE_CANDIDATES:
        errors.append("side_to_move candidate set mismatch")

    try:
        expected = _expected_answer(task, item["fen"], item.get("params", {}))
    except Exception as exc:  # noqa: BLE001 - 검증 실패 사유를 그대로 보고
        return errors + [f"recompute failed: {exc}"]
    if expected != item.get("answer"):
        errors.append(f"answer mismatch: item={item.get('answer')!r} recomputed={expected!r}")
    return errors


def verify_corpus(items: Iterable[dict[str, Any]], max_error_samples: int = 20) -> dict[str, Any]:
    """전량 검증 + 분포 집계. mismatches==0 이어야 P0 통과."""
    n = 0
    n_bad = 0
    error_samples: list[dict[str, Any]] = []
    class_counts: dict[str, Counter] = {}
    for item in items:
        n += 1
        errs = verify_item(item)
        if errs:
            n_bad += 1
            if len(error_samples) < max_error_samples:
                error_samples.append({"qa_id": item.get("qa_id"), "errors": errs})
        answer = str(item.get("answer"))
        if item["task"] == "king_square":
            cls = item.get("params", {}).get("color", "?")
        elif answer.startswith("there is no"):
            cls = "nosuch"
        else:
            cls = answer
        class_counts.setdefault(item["task"], Counter())[cls] += 1
    return {
        "n_items": n,
        "n_mismatched": n_bad,
        "error_samples": error_samples,
        "answer_distribution": {t: dict(c) for t, c in class_counts.items()},
    }
