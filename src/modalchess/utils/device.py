"""디바이스 및 autocast 유틸리티."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Iterator

import torch


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """CUDA를 우선으로 하는 기본 실행 디바이스를 반환한다."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_autocast_dtype(device: torch.device) -> torch.dtype | None:
    """주어진 디바이스에서 안전한 autocast dtype을 고른다."""
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device) -> Iterator[object]:
    """CUDA에서 실행 중이면 autocast 컨텍스트 매니저를 반환한다."""
    dtype = resolve_autocast_dtype(device)
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dtype)
