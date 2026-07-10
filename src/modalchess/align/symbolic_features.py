"""Hybrid 텍스트 표현용 심볼릭 특징: board(수 정체성) + text(move-mention 파싱).

Phase 1 진단 ①의 결론(병목 = MiniLM이 move 토큰 식별 정보를 버림)에 대한 1순위 레버.
- board 쪽: pair의 정의가 (position, target_move) ↔ comment이므로 target move의
  from/to/기물/플래그를 심볼릭 벡터로 만든다. 정답 누출이 아니라 pair 키의 명시화다.
- text 쪽: 코멘트 원문에서만 파싱한 SAN/UCI mention 특징. board 쪽 정보는 쓰지 않는다.
- connector가 두 채널의 정렬을 학습한다(문장 임베딩에 concat).

정직 캐비엇: 이 특징으로 오르는 retrieval은 심볼릭 신호의 회수이지 언어 이해의 증명이
아니다. lowercase SAN 파싱은 모호성이 있다(예: "bxc3"의 b가 비숍인지 b파일 폰인지) —
multi-hot에 그대로 담고 가중치는 connector가 학습한다.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import chess
import torch

from modalchess.align.oracle_ceiling import move_attributes
from modalchess.align.text_embed import normalize_comment

BOARD_FEATURE_DIM = 140
TEXT_FEATURE_DIM = 333

_PIECE_ORDER = "pnbrqk"  # chess.PAWN(1) .. chess.KING(6) → slot 0..5

_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
_SAN_RE = re.compile(r"^([kqrbn])?([a-h])?([1-8])?(x)?([a-h][1-8])(?:=?([qrbn]))?$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def board_move_features(fen: str, uci: str) -> torch.Tensor:
    """(fen, target_move)의 심볼릭 벡터 [140].

    [0:64] from one-hot, [64:128] to one-hot, [128:134] piece one-hot,
    [134:140] is_capture/gives_check/is_mate/is_promotion/is_castling/side_white.
    비합법 수면 zero 벡터.
    """
    out = torch.zeros(BOARD_FEATURE_DIM)
    attrs = move_attributes(fen, uci)
    if attrs is None:
        return out
    move = chess.Move.from_uci(uci)
    out[move.from_square] = 1.0
    out[64 + move.to_square] = 1.0
    if 1 <= attrs["piece_type"] <= 6:
        out[128 + attrs["piece_type"] - 1] = 1.0
    flags = (
        attrs["is_capture"], attrs["gives_check"], attrs["is_mate"],
        attrs["is_promotion"], attrs["is_castling"], attrs["side_to_move"],
    )
    for offset, flag in enumerate(flags):
        out[134 + offset] = float(flag)
    return out


def _square_index(name: str) -> int:
    return chess.parse_square(name)


def text_mention_features(normalized_text: str) -> torch.Tensor:
    """코멘트 원문에서 파싱한 move-mention 벡터 [333].

    [0:64]    UCI mention from multi-hot (모든 언급)
    [64:128]  UCI mention to multi-hot
    [128:192] 첫 UCI mention from one-hot (전술 라인의 첫 수 = 대개 target move)
    [192:256] 첫 UCI mention to one-hot
    [256:320] SAN mention 목적지 multi-hot (bare square 언급 포함 — 가중치는 학습에 위임)
    [320:326] SAN 기물 multi-hot (p,n,b,r,q,k)
    [326:331] 마커: capture(x)/check(+·단어)/mate(#·단어)/castle-kingside/castle-queenside
    [331]     has_uci_mention, [332] has_san_mention
    """
    out = torch.zeros(TEXT_FEATURE_DIM)
    text = normalized_text
    tokens = _TOKEN_RE.findall(text)

    first_uci_seen = False
    has_uci = False
    has_san = False
    for token in tokens:
        if _UCI_RE.fullmatch(token):
            has_uci = True
            from_idx = _square_index(token[0:2])
            to_idx = _square_index(token[2:4])
            out[from_idx] = 1.0
            out[64 + to_idx] = 1.0
            if not first_uci_seen:
                out[128 + from_idx] = 1.0
                out[192 + to_idx] = 1.0
                first_uci_seen = True
            continue
        match = _SAN_RE.fullmatch(token)
        if match:
            piece, _file, _rank, capture, dest, promo = match.groups()
            has_san = True
            out[256 + _square_index(dest)] = 1.0
            piece_letter = piece or "p"
            out[320 + _PIECE_ORDER.index(piece_letter)] = 1.0
            if capture:
                out[326] = 1.0
            if promo:
                out[320 + _PIECE_ORDER.index(promo)] = 1.0

    # capture 마커는 SAN 파싱에서만 세운다 (일반 단어의 x 오검출 방지)
    if "+" in text or re.search(r"\bcheck\b", text):
        out[327] = 1.0
    if "#" in text or re.search(r"\b(checkmate|mate)\b", text):
        out[328] = 1.0
    if re.search(r"(?:^|[^o0-])(?:o-o-o|0-0-0)(?:$|[^-])", text):
        out[330] = 1.0
    elif re.search(r"(?:^|[^o0-])(?:o-o|0-0)(?:$|[^-])", text):
        out[329] = 1.0
    out[331] = float(has_uci)
    out[332] = float(has_san)
    return out


def precompute_symbolic_features(
    corpus_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """corpus JSONL 전 행의 board/text 심볼릭 특징을 계산해 .pt로 저장."""
    probe_ids: list[str] = []
    board_rows: list[torch.Tensor] = []
    text_rows: list[torch.Tensor] = []
    with open(corpus_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            probe_ids.append(str(row["probe_id"]))
            board_rows.append(board_move_features(str(row["fen"]), str(row["target_move_uci"])))
            text_rows.append(text_mention_features(normalize_comment(str(row.get("comment_text", "")))))
    payload = {
        "feature_version": "sym_v1",
        "probe_id": probe_ids,
        "board_features": torch.stack(board_rows),
        "text_features": torch.stack(text_rows),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "n_rows": len(probe_ids),
        "board_dim": BOARD_FEATURE_DIM,
        "text_dim": TEXT_FEATURE_DIM,
        "output": str(output),
    }
