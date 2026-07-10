"""oracle_ceiling의 상한/하한 산식 정합성 테스트."""

from __future__ import annotations

import math

import pytest
import torch

from modalchess.align.oracle_ceiling import (
    assignment_ceiling,
    duplicate_ceiling,
    mention_score_matrix,
    metrics_from_scores,
    move_attributes,
    symbolic_score_matrix,
    tie_ceiling,
)

KS = (1, 5, 10, 50)


def test_assignment_ceiling_duplicate_group() -> None:
    # 같은 텍스트 2행 + 유일 텍스트 1행: 그룹 {1,2} rank + rank 1
    out = assignment_ceiling(["a", "a", "b"], ks=KS)
    expected_mrr = ((1.0 + 0.5) + 1.0) / 3
    assert out["mrr"] == pytest.approx(expected_mrr)
    assert out["r@1"] == pytest.approx((1 + 1) / 3)  # 그룹에서 1행, 유일 1행
    assert out["r@5"] == pytest.approx(1.0)


def test_assignment_ceiling_all_unique_is_perfect() -> None:
    out = assignment_ceiling(["a", "b", "c"], ks=KS)
    assert out["mrr"] == pytest.approx(1.0)
    assert out["r@1"] == pytest.approx(1.0)


def test_tie_ceiling_strict_rank() -> None:
    out = tie_ceiling([1, 2, 4], ks=KS)
    assert out["mrr"] == pytest.approx((1.0 + 0.5 + 0.25) / 3)
    assert out["r@1"] == pytest.approx(1 / 3)
    assert out["r@5"] == pytest.approx(1.0)


def test_duplicate_ceiling_combined_is_min() -> None:
    out = duplicate_ceiling(["a", "a", "b"], ["x", "x", "y"], ks=KS)
    for name in ("mrr", "r@1"):
        assert out["combined_min"][name] == pytest.approx(
            min(out["assignment"][name], out["tie"][name])
        )


def test_move_attributes_basics() -> None:
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    attrs = move_attributes(start, "e2e4")
    assert attrs is not None
    assert attrs["san"] == "e4"
    assert attrs["piece_type"] == 1  # pawn
    assert not attrs["is_capture"] and not attrs["gives_check"]
    # 비합법 수는 None
    assert move_attributes(start, "e2e5") is None


def test_move_attributes_mate_flags() -> None:
    # scholar's mate 직전: Qxf7#
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    attrs = move_attributes(fen, "f3f7")
    assert attrs is not None
    assert attrs["is_capture"] and attrs["gives_check"] and attrs["is_mate"]


def test_symbolic_uci_exact_ties() -> None:
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    attrs = [
        move_attributes(start, "e2e4"),
        move_attributes(start, "e2e4"),
        move_attributes(start, "g1f3"),
    ]
    scores = symbolic_score_matrix(attrs, "uci_exact")
    result = metrics_from_scores(scores, ks=KS)
    t2b = result["ranks"]["text_to_board"]
    # 행 0/1은 서로 동점(같은 uci) → strict rank 2, 행 2는 유일 → rank 1
    assert t2b.tolist() == [2.0, 2.0, 1.0]


def test_symbolic_move_dominates_flags() -> None:
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    attrs = [
        move_attributes(start, "e2e4"),
        move_attributes(start, "d2d4"),  # 같은 플래그(폰 전진), 다른 수
    ]
    scores = symbolic_score_matrix(attrs, "move_plus_flags")
    assert scores[0, 0] > scores[0, 1]  # move 일치가 flag 전체 일치보다 크다


def test_symbolic_none_attrs_never_match() -> None:
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    attrs = [None, None, move_attributes(start, "e2e4")]
    scores = symbolic_score_matrix(attrs, "uci_exact")
    assert scores[0, 1] == 0.0 and scores[1, 0] == 0.0
    assert scores[0, 0] == 1.0  # sentinel은 자기 자신과만 일치


def test_mention_matrix_uci_and_san() -> None:
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    attrs = [move_attributes(start, "g1f3"), move_attributes(start, "e2e4")]
    comments = [
        "develop the knight with nf3 toward the center",  # SAN 언급
        "tactic e2e4 e7e5 continues",  # UCI 언급
    ]
    scores = mention_score_matrix(comments, attrs)
    assert scores[0, 0] == 1.0  # nf3 토큰이 SAN과 일치
    assert scores[1, 1] == 1.0  # e2e4 토큰이 UCI와 일치
    assert scores[0, 1] == 0.0
    # 부분 문자열 오검출 방지: "be4"는 e4 언급이 아니다
    scores2 = mention_score_matrix(["the bishop lands on be4 square"], [attrs[1]])
    assert scores2[0, 0] == 0.0


def test_mention_castling() -> None:
    fen = "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    attrs = [move_attributes(fen, "e1g1")]
    assert attrs[0] is not None and attrs[0]["is_castling"]
    scores = mention_score_matrix(["white castles with o-o for safety"], attrs)
    assert scores[0, 0] == 1.0
