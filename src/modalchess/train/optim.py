"""옵티마이저와 LR 스케줄러 유틸리티."""

from __future__ import annotations

import math
from typing import Any

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """베이스라인 학습용 기본 옵티마이저를 만든다."""
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def warmup_cosine_lr_lambda(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    """warmup 후 cosine으로 감쇠하는 step별 LR 배율을 계산한다."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_lr_scheduler(
    optimizer: AdamW,
    schedule_config: dict[str, Any] | None,
    total_steps: int,
) -> LRScheduler | None:
    """config 기반 step 단위 LR 스케줄러를 만든다. constant면 None을 반환한다."""
    config = schedule_config or {}
    name = str(config.get("name", "constant"))
    if name == "constant":
        return None
    if name == "warmup_cosine":
        if total_steps <= 0:
            raise ValueError("warmup_cosine 스케줄러에는 total_steps > 0이 필요하다.")
        warmup_ratio = float(config.get("warmup_ratio", 0.03))
        min_lr_ratio = float(config.get("min_lr_ratio", 0.05))
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio는 [0, 1) 범위여야 한다: {warmup_ratio}")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio는 [0, 1] 범위여야 한다: {min_lr_ratio}")
        warmup_steps = int(total_steps * warmup_ratio)
        return LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_cosine_lr_lambda(
                step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=min_lr_ratio,
            ),
        )
    raise ValueError(f"지원하지 않는 lr schedule: {name}")
