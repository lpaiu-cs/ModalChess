"""ModalChess spatial baseline용 손실 함수 조합."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _cross_entropy_or_zero(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if (targets != -100).sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits, targets, ignore_index=-100)


def compute_modalchess_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    """베이스라인 모델에 필요한 손실을 설정값에 따라 조합한다."""
    src_loss = _cross_entropy_or_zero(outputs["src_logits"], batch["src_targets"])
    dst_loss = _cross_entropy_or_zero(outputs["dst_logits"], batch["dst_targets"])
    promo_loss = _cross_entropy_or_zero(outputs["promo_logits"], batch["promo_targets"])
    policy_loss = (src_loss + dst_loss + promo_loss) / 3.0

    piece_loss = F.binary_cross_entropy_with_logits(
        outputs["piece_logits"],
        batch["state_piece_planes"],
    )
    side_to_move_loss = F.binary_cross_entropy_with_logits(
        outputs["side_to_move_logits"],
        batch["state_side_to_move"],
    )
    castling_loss = F.binary_cross_entropy_with_logits(
        outputs["castling_logits"],
        batch["state_castling_rights"],
    )
    en_passant_loss = F.cross_entropy(
        outputs["en_passant_logits"],
        batch["state_en_passant"],
    )
    in_check_loss = F.binary_cross_entropy_with_logits(
        outputs["in_check_logits"],
        batch["state_in_check"],
    )
    state_probe_loss = (
        piece_loss + side_to_move_loss + castling_loss + en_passant_loss + in_check_loss
    ) / 5.0

    legality_loss = F.binary_cross_entropy_with_logits(
        outputs["legality_logits"],
        batch["legality_matrix"],
    )
    value_loss = F.mse_loss(outputs["value_logits"], batch["value_targets"])
    concept_loss = F.binary_cross_entropy_with_logits(
        outputs["concept_logits"],
        batch["concept_targets"],
    )

    total_loss = (
        weights.get("policy", 1.0) * policy_loss
        + weights.get("state_probe", 1.0) * state_probe_loss
        + weights.get("legality", 1.0) * legality_loss
        + weights.get("value", 1.0) * value_loss
        + weights.get("concept", 1.0) * concept_loss
    )

    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "state_probe_loss": state_probe_loss,
        "legality_loss": legality_loss,
        "value_loss": value_loss,
        "concept_loss": concept_loss,
    }
