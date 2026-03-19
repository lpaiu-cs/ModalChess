"""미래 move-conditioned rationale 모듈을 위한 최소 인터페이스."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RationaleRequest:
    """미래 rationale 생성 모듈을 위한 자리표시자 요청 객체."""

    fen: str
    move_uci: str
    context: dict[str, str] | None = None


class FutureRationaleInterface:
    """향후 rationale 서브시스템을 위해 남겨둔 스텁 인터페이스."""

    def generate(self, request: RationaleRequest) -> str:
        raise NotImplementedError("Rationale generation is out of scope for this phase.")
