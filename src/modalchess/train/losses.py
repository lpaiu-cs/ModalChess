"""ModalChess spatial baseline용 손실 함수 조합."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from modalchess.models.heads.policy_factorized import score_factorized_moves


def _cross_entropy_or_zero(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if (targets != -100).sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits, targets, ignore_index=-100)


def _listwise_policy_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    target_indices = batch["target_legal_move_index"]
    for sample_index, legal_moves in enumerate(batch["legal_moves_factorized"]):
        target_index = int(target_indices[sample_index].item())
        if target_index < 0 or not legal_moves:
            continue
        sample_outputs = {
            "src_logits": outputs["src_logits"][sample_index],
            "dst_logits": outputs["dst_logits"][sample_index],
            "promo_logits": outputs["promo_logits"][sample_index],
        }
        if "pair_logits" in outputs:
            sample_outputs["pair_logits"] = outputs["pair_logits"][sample_index]
        scores = score_factorized_moves(sample_outputs, legal_moves)
        losses.append(
            F.cross_entropy(
                scores.unsqueeze(0),
                torch.tensor([target_index], dtype=torch.long, device=scores.device),
            )
        )
    if not losses:
        return outputs["src_logits"].sum() * 0.0
    return torch.stack(losses).mean()


def _weighted_legality_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_pos_weight: float,
) -> torch.Tensor:
    positives = targets.sum()
    negatives = targets.numel() - positives
    if positives.item() == 0:
        pos_weight = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)
    else:
        pos_weight = torch.clamp(negatives / positives, min=1.0, max=max_pos_weight)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def _masked_mse_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(logits, targets)
    active = mask.to(logits.device).bool()
    if active.sum() == 0:
        return logits.sum() * 0.0
    return F.mse_loss(logits[active], targets[active])


def _masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return F.binary_cross_entropy_with_logits(logits, targets)
    active = mask.to(logits.device).bool()
    if active.sum() == 0:
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    expanded_mask = active.unsqueeze(-1).expand_as(losses)
    return (losses * expanded_mask).sum() / expanded_mask.sum().clamp_min(1)


def compute_modalchess_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    """베이스라인 모델에 필요한 손실을 설정값에 따라 조합한다."""
    src_loss = _cross_entropy_or_zero(outputs["src_logits"], batch["src_targets"])
    dst_loss = _cross_entropy_or_zero(outputs["dst_logits"], batch["dst_targets"])
    promo_loss = _cross_entropy_or_zero(outputs["promo_logits"], batch["promo_targets"])
    axis_policy_loss = (src_loss + dst_loss + promo_loss) / 3.0
    listwise_policy_loss = _listwise_policy_loss(outputs, batch)
    axis_weight = weights.get("policy_axis_ce", 1.0)
    listwise_weight = weights.get("policy_listwise", 1.0)
    policy_loss = (
        axis_weight * axis_policy_loss + listwise_weight * listwise_policy_loss
    ) / max(axis_weight + listwise_weight, 1e-8)

    square_state_loss = F.cross_entropy(
        outputs["square_state_logits"],
        batch["state_square_classes"],
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
        square_state_loss + side_to_move_loss + castling_loss + en_passant_loss + in_check_loss
    ) / 5.0

    legality_loss = _weighted_legality_loss(
        outputs["legality_logits"],
        batch["legality_tensor"],
        max_pos_weight=float(weights.get("legality_pos_weight_cap", 64.0)),
    )
    value_loss = _masked_mse_loss(
        outputs["value_logits"],
        batch["value_targets"],
        batch.get("has_engine_eval"),
    )
    concept_loss = _masked_bce_with_logits(
        outputs["concept_logits"],
        batch["concept_targets"],
        batch.get("has_concept_labels"),
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
        "policy_axis_loss": axis_policy_loss,
        "policy_listwise_loss": listwise_policy_loss,
        "state_probe_loss": state_probe_loss,
        "legality_loss": legality_loss,
        "value_loss": value_loss,
        "concept_loss": concept_loss,
    }
