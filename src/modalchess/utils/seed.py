"""재현 가능한 시드 설정 유틸리티."""

from __future__ import annotations

import random

import torch


def seed_everything(seed: int) -> None:
    """Python과 Torch RNG를 함께 시드한다."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
