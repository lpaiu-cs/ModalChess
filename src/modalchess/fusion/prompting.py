"""ChatML 프롬프트 조립 — 전 arm 공통 골격, 보드 표현 자리만 교체 (docs/phase2_plan.md §3).

세그먼트: [PRE] [board 표현: 주입 임베딩 64개 또는 "FEN: ..." 텍스트] [POST+question] [answer]
전 arm이 동일 텍스트 골격을 공유해야 델타 해석이 성립한다.
"""

from __future__ import annotations

SYSTEM_PROMPT = "You are a chess assistant. Look at the given board and answer with the single best option."

PRE_BOARD = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nBoard:\n"


def post_board(question: str) -> str:
    """보드 표현 뒤 ~ assistant 응답 직전까지."""
    return f"\n{question}<|im_end|>\n<|im_start|>assistant\n"


def answer_segment(answer: str) -> str:
    return f"{answer}<|im_end|>"


def fen_board_text(fen: str) -> str:
    """FEN arm의 보드 표현(주입 임베딩 대신 텍스트)."""
    return f"FEN: {fen}"
