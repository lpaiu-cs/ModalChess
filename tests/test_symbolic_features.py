"""symbolic_features의 파싱·벡터 레이아웃 정합성 테스트."""

from __future__ import annotations

import chess
import pytest
import torch

from modalchess.align.symbolic_features import (
    BOARD_FEATURE_DIM,
    TEXT_FEATURE_DIM,
    board_move_features,
    text_mention_features,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_board_features_layout() -> None:
    out = board_move_features(START, "e2e4")
    assert out.shape == (BOARD_FEATURE_DIM,)
    assert out[chess.E2] == 1.0                     # from
    assert out[64 + chess.E4] == 1.0                # to
    assert out[128 + chess.PAWN - 1] == 1.0         # piece
    assert out[134] == 0.0 and out[135] == 0.0      # no capture/check
    assert out[139] == 1.0                          # white to move


def test_board_features_mate_flags() -> None:
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    out = board_move_features(fen, "f3f7")
    assert out[134] == 1.0 and out[135] == 1.0 and out[136] == 1.0  # capture+check+mate


def test_board_features_illegal_is_zero() -> None:
    assert board_move_features(START, "e2e5").abs().sum() == 0.0


def test_text_features_uci_first_mention() -> None:
    out = text_mention_features("tactic d3h7 g8f8 h7h8 checkmate")
    assert out.shape == (TEXT_FEATURE_DIM,)
    # 모든 UCI 언급의 from/to multi-hot
    assert out[chess.D3] == 1.0 and out[chess.G8] == 1.0 and out[chess.H7] == 1.0
    assert out[64 + chess.H7] == 1.0 and out[64 + chess.F8] == 1.0
    # 첫 언급 one-hot = d3h7만
    assert out[128 + chess.D3] == 1.0 and out[128 + chess.G8] == 0.0
    assert out[192 + chess.H7] == 1.0 and out[192 + chess.F8] == 0.0
    assert out[328] == 1.0  # mate 단어
    assert out[331] == 1.0 and out[332] == 0.0


def test_text_features_san() -> None:
    out = text_mention_features("develop with nf3 then bxc3 wins material")
    assert out[256 + chess.F3] == 1.0 and out[256 + chess.C3] == 1.0  # SAN 목적지
    assert out[320 + 1] == 1.0  # knight (p,n,b,r,q,k에서 n=slot1)
    assert out[326] == 1.0      # capture 마커
    assert out[332] == 1.0 and out[331] == 0.0


def test_text_features_castling() -> None:
    kingside = text_mention_features("white castles o-o for safety")
    queenside = text_mention_features("black goes long with o-o-o here")
    assert kingside[329] == 1.0 and kingside[330] == 0.0
    assert queenside[330] == 1.0 and queenside[329] == 0.0


def test_text_features_no_mention_is_sparse() -> None:
    out = text_mention_features("a very generic comment about strategy")
    assert out[331] == 0.0 and out[332] == 0.0
    # 첫-언급 슬롯은 완전 0
    assert out[128:256].abs().sum() == 0.0


def test_text_features_word_not_uci() -> None:
    # 일반 단어는 UCI로 오검출되지 않는다
    out = text_mention_features("the best defense here")
    assert out[331] == 0.0
