"""ModalChess 상태 충실도 평가 지표."""

from __future__ import annotations

import torch

from modalchess.data.tensor_codec import PIECE_CHANNELS


def _count_exact_match(
    predicted_square_state: torch.Tensor,
    target_square_state: torch.Tensor,
) -> torch.Tensor:
    matches = []
    for class_index in range(1, len(PIECE_CHANNELS) + 1):
        pred_count = (predicted_square_state == class_index).sum(dim=(1, 2))
        target_count = (target_square_state == class_index).sum(dim=(1, 2))
        matches.append(pred_count == target_count)
    return torch.stack(matches, dim=0).all(dim=0).float().mean()


def _king_count_validity(predicted_square_state: torch.Tensor) -> torch.Tensor:
    white_king_valid = (predicted_square_state == 6).sum(dim=(1, 2)) == 1
    black_king_valid = (predicted_square_state == 12).sum(dim=(1, 2)) == 1
    return (white_king_valid & black_king_valid).float().mean()


def _binary_precision_recall_f1(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    true_positive = (predicted & target).sum().float()
    false_positive = (predicted & ~target).sum().float()
    false_negative = (~predicted & target).sum().float()
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return precision, recall, f1


def _average_precision(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    flat_scores = scores.reshape(-1)
    flat_targets = targets.reshape(-1).float()
    positives = flat_targets.sum()
    if positives.item() == 0:
        return torch.tensor(0.0, device=scores.device)
    ranking = torch.argsort(flat_scores, descending=True)
    ranked_targets = flat_targets[ranking]
    cumulative_hits = torch.cumsum(ranked_targets, dim=0)
    precision = cumulative_hits / torch.arange(
        1,
        ranked_targets.numel() + 1,
        device=scores.device,
        dtype=torch.float32,
    )
    return (precision * ranked_targets).sum() / positives


def _recall_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    recalls = []
    flat_scores = scores.view(scores.size(0), -1)
    flat_targets = targets.view(targets.size(0), -1).bool()
    for sample_scores, sample_targets in zip(flat_scores, flat_targets):
        positives = sample_targets.sum()
        if positives.item() == 0:
            continue
        topk = min(k, sample_scores.numel())
        indices = torch.topk(sample_scores, k=topk).indices
        recalls.append(sample_targets[indices].float().sum() / positives.float())
    if not recalls:
        return torch.tensor(0.0, device=scores.device)
    return torch.stack(recalls).mean()


def _subset_legality_recall(
    predicted: torch.Tensor,
    target: torch.Tensor,
    subset_mask: torch.Tensor,
) -> torch.Tensor:
    if subset_mask.sum().item() == 0:
        return torch.tensor(0.0, device=predicted.device)
    subset_pred = predicted[subset_mask]
    subset_target = target[subset_mask]
    true_positive = (subset_pred & subset_target).sum().float()
    false_negative = (~subset_pred & subset_target).sum().float()
    return true_positive / (true_positive + false_negative).clamp_min(1.0)


def compute_state_fidelity_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """복원 기반 상태 충실도 지표를 계산한다."""
    predicted_square_state = outputs["square_state_logits"].argmax(dim=1)
    target_square_state = batch["state_square_classes"]
    square_state_accuracy = (predicted_square_state == target_square_state).float().mean()
    piece_count_exact_match = _count_exact_match(predicted_square_state, target_square_state)
    king_count_validity = _king_count_validity(predicted_square_state)
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
    legality_scores = torch.sigmoid(outputs["legality_logits"])
    legality_predicted = legality_scores > 0.5
    legality_target = batch["legality_tensor"] > 0.5
    legality_precision, legality_recall, legality_f1 = _binary_precision_recall_f1(
        legality_predicted,
        legality_target,
    )
    legality_average_precision = _average_precision(legality_scores, legality_target.float())
    legality_recall_at_16 = _recall_at_k(legality_scores, legality_target.float(), k=16)
    promotion_legality_recall = _subset_legality_recall(
        legality_predicted,
        legality_target,
        batch["subset_promotion"],
    )
    castling_legality_recall = _subset_legality_recall(
        legality_predicted,
        legality_target,
        batch["subset_castling"],
    )
    en_passant_legality_recall = _subset_legality_recall(
        legality_predicted,
        legality_target,
        batch["subset_en_passant"],
    )
    check_evasion_legality_recall = _subset_legality_recall(
        legality_predicted,
        legality_target,
        batch["subset_check_evasion"],
    )
    return {
        "square_state_accuracy": float(square_state_accuracy.detach().cpu()),
        "piece_count_exact_match": float(piece_count_exact_match.detach().cpu()),
        "king_count_validity": float(king_count_validity.detach().cpu()),
        "side_to_move_accuracy": float(side_to_move_accuracy.detach().cpu()),
        "castling_right_accuracy": float(castling_accuracy.detach().cpu()),
        "en_passant_accuracy": float(en_passant_accuracy.detach().cpu()),
        "in_check_accuracy": float(in_check_accuracy.detach().cpu()),
        "legality_precision": float(legality_precision.detach().cpu()),
        "legality_recall": float(legality_recall.detach().cpu()),
        "legality_f1": float(legality_f1.detach().cpu()),
        "legality_average_precision": float(legality_average_precision.detach().cpu()),
        "legality_recall_at_16": float(legality_recall_at_16.detach().cpu()),
        "promotion_legality_recall": float(promotion_legality_recall.detach().cpu()),
        "castling_legality_recall": float(castling_legality_recall.detach().cpu()),
        "en_passant_legality_recall": float(en_passant_legality_recall.detach().cpu()),
        "check_evasion_legality_recall": float(check_evasion_legality_recall.detach().cpu()),
    }
