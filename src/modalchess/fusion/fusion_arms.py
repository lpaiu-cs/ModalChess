"""P1 arm 모듈: Board / RawBoard / Blind / FEN-soft / FEN-zs (docs/phase2_plan.md §3).

전 arm 동일 기판(frozen LM) 위에서 보드 표현 채널만 교체한다. 학습 파라미터는
projection MLP 또는 soft 토큰뿐이며, LM과 보드 백본은 동결.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from modalchess.data.fen_codec import fen_to_board_state
from modalchess.data.tensor_codec import encode_fen_history
from modalchess.train.train_spatial_baseline import build_model_from_config
from modalchess.utils.square_utils import square_to_coords

ARM_KINDS = ("board", "rawboard", "blind", "fen_soft", "fen_zs")


def fen_to_planes_meta(fen: str, history_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """FEN → (board_planes [H,C,8,8], meta [3]) — 백본 사전학습과 동일 전처리."""
    planes = encode_fen_history([fen], history_length)
    state = fen_to_board_state(fen)
    meta = torch.tensor(
        [
            float(state.meta.halfmove_clock),
            float(state.meta.fullmove_number),
            float(state.meta.repetition_count),
        ],
        dtype=torch.float32,
    )
    return planes, meta


def raw_square_features(planes: torch.Tensor) -> torch.Tensor:
    """[B,H,C,8,8] → [B,64,H*C] — square 인덱스(a1=0..h8=63) 순서, 인코더와 동일 좌표계."""
    batch, history, channels, _, _ = planes.shape
    flat = planes.reshape(batch, history * channels, 64)
    idx = torch.tensor(
        [row * 8 + col for row, col in (square_to_coords(s) for s in range(64))],
        dtype=torch.long, device=planes.device,
    )
    flat = flat.index_select(-1, idx)
    return flat.permute(0, 2, 1)


class FrozenBoardBackbone(nn.Module):
    """G3 체크포인트에서 meta_encoder+encoder만 취해 64칸 토큰을 낸다 (동결)."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = checkpoint["resolved_model_config"]
        core = build_model_from_config(model_config)
        core.load_state_dict(checkpoint["model_state_dict"])
        self.meta_encoder = core.meta_encoder
        self.encoder = core.encoder
        self.history_length = int(model_config["history_length"])
        self.d_model = int(model_config["d_model"])
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, board_planes: torch.Tensor, meta_features: torch.Tensor) -> torch.Tensor:
        extra = self.meta_encoder(meta_features)
        return self.encoder(board_planes, extra_tokens=extra)["tokens"]


class RMSScale(nn.Module):
    """토큰별 RMS 정규화 후 학습형 스칼라(초기값 = LM 임베딩 RMS 캘리브레이션)로 스케일."""

    def __init__(self, init_scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return x / rms * self.scale


class ProjectionMLP(nn.Module):
    """LN → Linear → GELU → Linear → RMSScale (2층, docs/phase2_plan.md §3)."""

    def __init__(self, d_in: int, d_lm: int, hidden: int, calib_rms: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_lm),
        )
        self.rms_scale = RMSScale(calib_rms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rms_scale(self.net(x))


class FusionArm(nn.Module):
    """보드 표현 채널. injected() → [B,64,d_lm] 또는 None(fen_zs)."""

    def __init__(
        self,
        kind: str,
        d_lm: int,
        calib_rms: float,
        proj_hidden: int,
        backbone: FrozenBoardBackbone | None = None,
        raw_dim: int = 18,
    ) -> None:
        super().__init__()
        if kind not in ARM_KINDS:
            raise ValueError(f"unknown arm kind: {kind}")
        self.kind = kind
        self.backbone = backbone
        if kind == "board":
            if backbone is None:
                raise ValueError("board arm needs a backbone")
            self.projection = ProjectionMLP(backbone.d_model, d_lm, proj_hidden, calib_rms)
        elif kind == "rawboard":
            self.projection = ProjectionMLP(raw_dim, d_lm, proj_hidden, calib_rms)
        elif kind in ("blind", "fen_soft"):
            self.soft_tokens = nn.Parameter(torch.randn(64, d_lm) * calib_rms)

    @property
    def uses_fen_text(self) -> bool:
        return self.kind in ("fen_soft", "fen_zs")

    @property
    def uses_board_planes(self) -> bool:
        return self.kind in ("board", "rawboard")

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("backbone.")]

    def injected(self, batch: dict[str, Any], device: torch.device) -> torch.Tensor | None:
        if self.kind == "fen_zs":
            return None
        if self.kind in ("blind", "fen_soft"):
            batch_size = batch["n"]
            return self.soft_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        planes = batch["planes"].to(device)
        if self.kind == "board":
            meta = batch["meta"].to(device)
            tokens = self.backbone(planes, meta)
            return self.projection(tokens)
        return self.projection(raw_square_features(planes))
