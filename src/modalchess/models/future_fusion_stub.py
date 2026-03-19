"""미래 LLM fusion 모듈을 위한 최소 인터페이스."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class FusionAdapterOutput:
    """미래 멀티모달 fusion 어댑터를 위한 자리표시자 출력."""

    fused_tokens: torch.Tensor
    auxiliary: dict[str, torch.Tensor] | None = None


class FutureFusionInterface(nn.Module):
    """향후 LLM fusion 단계를 위해 남겨둔 스텁 인터페이스."""

    def forward(
        self,
        board_tokens: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
    ) -> FusionAdapterOutput:
        raise NotImplementedError("LLM fusion is out of scope for this phase.")
