"""ModalChess 상태 충실도 평가 지표."""

from __future__ import annotations

import torch


def compute_state_fidelity_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """복원 기반 상태 충실도 지표를 계산한다."""
    piece_accuracy = (
        (torch.sigmoid(outputs["piece_logits"]) > 0.5) == (batch["state_piece_planes"] > 0.5)
    ).float().mean()
    side_to_move_accuracy = (
        (torch.sigmoid(outputs["side_to_move_logits"]) > 0.5)
        == (batch["state_side_to_move"] > 0.5)
    ).float().mean()
    castling_accuracy = (
        (torch.sigmoid(outputs["castling_logits"]) > 0.5)
        == (batch["state_castling_rights"] > 0.5)
    ).float().mean()
    en_passant_accuracy = (
        outputs["en_passant_logits"].argmax(dim=-1) == batch["state_en_passant"]
    ).float().mean()
    in_check_accuracy = (
        (torch.sigmoid(outputs["in_check_logits"]) > 0.5)
        == (batch["state_in_check"] > 0.5)
    ).float().mean()
    legality_accuracy = (
        (torch.sigmoid(outputs["legality_logits"]) > 0.5) == (batch["legality_matrix"] > 0.5)
    ).float().mean()
    return {
        "piece_occupancy_accuracy": float(piece_accuracy.detach().cpu()),
        "side_to_move_accuracy": float(side_to_move_accuracy.detach().cpu()),
        "castling_right_accuracy": float(castling_accuracy.detach().cpu()),
        "en_passant_accuracy": float(en_passant_accuracy.detach().cpu()),
        "in_check_accuracy": float(in_check_accuracy.detach().cpu()),
        "legality_accuracy": float(legality_accuracy.detach().cpu()),
    }
