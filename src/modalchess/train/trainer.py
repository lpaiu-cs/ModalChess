"""ModalChess 베이스라인용 CUDA-aware 트레이너."""

from __future__ import annotations

from collections import defaultdict
from itertools import cycle
from typing import Any

import torch

from modalchess.train.losses import compute_modalchess_losses
from modalchess.utils.device import autocast_context, resolve_autocast_dtype, resolve_device


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """배치 내 텐서 값을 대상 디바이스로 옮긴다."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class Trainer:
    """선택형 CUDA autocast를 지원하는 최소 단일 디바이스 트레이너."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_weights: dict[str, float],
        device: torch.device | None = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        self.device = device or resolve_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.grad_clip_norm = grad_clip_norm
        autocast_dtype = resolve_autocast_dtype(self.device)
        use_grad_scaler = self.device.type == "cuda" and autocast_dtype == torch.float16
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """한 번의 최적화 스텝을 수행하고 스칼라 손실을 반환한다."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = move_batch_to_device(batch, self.device)
        with autocast_context(self.device):
            outputs = self.model(batch["board_planes"])
            losses = compute_modalchess_losses(outputs, batch, self.loss_weights)
        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(losses["total_loss"]).backward()
            if self.grad_clip_norm is not None:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses["total_loss"].backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
        return {name: float(loss.detach().cpu()) for name, loss in losses.items()}

    def train_epoch(self, dataloader) -> dict[str, float]:
        """한 epoch 동안 학습하고 평균 손실 지표를 계산한다."""
        totals: dict[str, float] = defaultdict(float)
        count = 0
        for batch in dataloader:
            step_losses = self.train_step(batch)
            for name, value in step_losses.items():
                totals[name] += value
            count += 1
        return {name: value / max(count, 1) for name, value in totals.items()}

    def overfit(self, dataloader, num_steps: int) -> dict[str, float]:
        """반복되는 dataloader 위에서 소형 overfit 루프를 수행한다."""
        iterator = cycle(dataloader)
        first_loss: float | None = None
        last_loss = 0.0
        for step in range(num_steps):
            step_losses = self.train_step(next(iterator))
            if step == 0:
                first_loss = step_losses["total_loss"]
            last_loss = step_losses["total_loss"]
        return {
            "initial_loss": float(first_loss or 0.0),
            "final_loss": float(last_loss),
            "num_steps": float(num_steps),
        }
