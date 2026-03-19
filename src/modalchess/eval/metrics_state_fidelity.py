"""ModalChess 상태 충실도 평가 지표."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from modalchess.data.tensor_codec import PIECE_CHANNELS


def _piece_count_match_mask(
    predicted_square_state: torch.Tensor,
    target_square_state: torch.Tensor,
) -> torch.Tensor:
    matches = []
    for class_index in range(1, len(PIECE_CHANNELS) + 1):
        pred_count = (predicted_square_state == class_index).sum(dim=(1, 2))
        target_count = (target_square_state == class_index).sum(dim=(1, 2))
        matches.append(pred_count == target_count)
    return torch.stack(matches, dim=0).all(dim=0)


def _king_count_valid_mask(predicted_square_state: torch.Tensor) -> torch.Tensor:
    white_king_valid = (predicted_square_state == 6).sum(dim=(1, 2)) == 1
    black_king_valid = (predicted_square_state == 12).sum(dim=(1, 2)) == 1
    return white_king_valid & black_king_valid


def _average_precision_weighted_sum(scores: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat_scores = scores.reshape(-1)
    flat_targets = targets.reshape(-1).float()
    positives = flat_targets.sum()
    if positives.item() == 0:
        zero = torch.tensor(0.0, device=scores.device)
        return zero, zero
    ranking = torch.argsort(flat_scores, descending=True)
    ranked_targets = flat_targets[ranking]
    cumulative_hits = torch.cumsum(ranked_targets, dim=0)
    precision = cumulative_hits / torch.arange(
        1,
        ranked_targets.numel() + 1,
        device=scores.device,
        dtype=torch.float32,
    )
    ap = (precision * ranked_targets).sum() / positives
    return ap * positives, positives


def _recall_at_k_sum_and_count(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    recall_sum = torch.tensor(0.0, device=scores.device)
    recall_count = torch.tensor(0.0, device=scores.device)
    flat_scores = scores.view(scores.size(0), -1)
    flat_targets = targets.view(targets.size(0), -1).bool()
    for sample_scores, sample_targets in zip(flat_scores, flat_targets):
        positives = sample_targets.sum()
        if positives.item() == 0:
            continue
        topk = min(k, sample_scores.numel())
        indices = torch.topk(sample_scores, k=topk).indices
        recall_sum = recall_sum + sample_targets[indices].float().sum() / positives.float()
        recall_count = recall_count + 1.0
    return recall_sum, recall_count


def _subset_recall_numerator_denominator(
    predicted: torch.Tensor,
    target: torch.Tensor,
    subset_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if subset_mask.sum().item() == 0:
        zero = torch.tensor(0.0, device=predicted.device)
        return zero, zero
    subset_pred = predicted[subset_mask]
    subset_target = target[subset_mask]
    true_positive = (subset_pred & subset_target).sum().float()
    denominator = subset_target.sum().float()
    return true_positive, denominator


@dataclass
class StateFidelityAccumulator:
    """배치 반복형 evaluation을 위한 상태 충실도 누적기."""

    square_correct: float = 0.0
    square_total: float = 0.0
    occupied_correct: float = 0.0
    occupied_total: float = 0.0
    piece_count_exact_match_sum: float = 0.0
    king_count_valid_sum: float = 0.0
    sample_count: float = 0.0
    side_correct: float = 0.0
    side_total: float = 0.0
    castling_correct: float = 0.0
    castling_total: float = 0.0
    en_passant_correct: float = 0.0
    en_passant_total: float = 0.0
    in_check_correct: float = 0.0
    in_check_total: float = 0.0
    legality_tp: float = 0.0
    legality_fp: float = 0.0
    legality_fn: float = 0.0
    legality_ap_weighted_sum: float = 0.0
    legality_positive_total: float = 0.0
    legality_recall_at_16_sum: float = 0.0
    legality_recall_at_16_count: float = 0.0
    promotion_recall_tp: float = 0.0
    promotion_recall_total: float = 0.0
    castling_recall_tp: float = 0.0
    castling_recall_total: float = 0.0
    en_passant_recall_tp: float = 0.0
    en_passant_recall_total: float = 0.0
    check_evasion_recall_tp: float = 0.0
    check_evasion_recall_total: float = 0.0
    piece_tp: torch.Tensor = field(default_factory=lambda: torch.zeros(len(PIECE_CHANNELS), dtype=torch.float32))
    piece_fp: torch.Tensor = field(default_factory=lambda: torch.zeros(len(PIECE_CHANNELS), dtype=torch.float32))
    piece_fn: torch.Tensor = field(default_factory=lambda: torch.zeros(len(PIECE_CHANNELS), dtype=torch.float32))
    non_empty_confusion: torch.Tensor = field(default_factory=lambda: torch.zeros(len(PIECE_CHANNELS), len(PIECE_CHANNELS) + 1, dtype=torch.long))

    def update(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        """현재 배치의 상태 충실도 통계를 누적한다."""
        predicted_square_state = outputs["square_state_logits"].argmax(dim=1)
        target_square_state = batch["state_square_classes"]
        occupied_mask = target_square_state != 0

        self.square_correct += float((predicted_square_state == target_square_state).sum().item())
        self.square_total += float(target_square_state.numel())
        self.occupied_correct += float(((predicted_square_state == target_square_state) & occupied_mask).sum().item())
        self.occupied_total += float(occupied_mask.sum().item())
        self.piece_count_exact_match_sum += float(_piece_count_match_mask(predicted_square_state, target_square_state).sum().item())
        self.king_count_valid_sum += float(_king_count_valid_mask(predicted_square_state).sum().item())
        self.sample_count += float(target_square_state.size(0))

        side_pred = torch.sigmoid(outputs["side_to_move_logits"]) > 0.5
        side_target = batch["state_side_to_move"] > 0.5
        self.side_correct += float((side_pred == side_target).sum().item())
        self.side_total += float(side_target.numel())

        castling_pred = torch.sigmoid(outputs["castling_logits"]) > 0.5
        castling_target = batch["state_castling_rights"] > 0.5
        self.castling_correct += float((castling_pred == castling_target).sum().item())
        self.castling_total += float(castling_target.numel())

        en_passant_pred = outputs["en_passant_logits"].argmax(dim=-1)
        en_passant_target = batch["state_en_passant"]
        self.en_passant_correct += float((en_passant_pred == en_passant_target).sum().item())
        self.en_passant_total += float(en_passant_target.numel())

        in_check_pred = torch.sigmoid(outputs["in_check_logits"]) > 0.5
        in_check_target = batch["state_in_check"] > 0.5
        self.in_check_correct += float((in_check_pred == in_check_target).sum().item())
        self.in_check_total += float(in_check_target.numel())

        legality_scores = torch.sigmoid(outputs["legality_logits"])
        legality_predicted = legality_scores > 0.5
        legality_target = batch["legality_tensor"] > 0.5
        self.legality_tp += float((legality_predicted & legality_target).sum().item())
        self.legality_fp += float((legality_predicted & ~legality_target).sum().item())
        self.legality_fn += float((~legality_predicted & legality_target).sum().item())

        ap_weighted_sum, positive_total = _average_precision_weighted_sum(
            legality_scores,
            legality_target.float(),
        )
        self.legality_ap_weighted_sum += float(ap_weighted_sum.detach().cpu())
        self.legality_positive_total += float(positive_total.detach().cpu())

        recall_sum, recall_count = _recall_at_k_sum_and_count(
            legality_scores,
            legality_target.float(),
            k=16,
        )
        self.legality_recall_at_16_sum += float(recall_sum.detach().cpu())
        self.legality_recall_at_16_count += float(recall_count.detach().cpu())

        for attr_prefix, subset_key in (
            ("promotion", "subset_promotion"),
            ("castling", "subset_castling"),
            ("en_passant", "subset_en_passant"),
            ("check_evasion", "subset_check_evasion"),
        ):
            numerator, denominator = _subset_recall_numerator_denominator(
                legality_predicted,
                legality_target,
                batch[subset_key],
            )
            setattr(self, f"{attr_prefix}_recall_tp", getattr(self, f"{attr_prefix}_recall_tp") + float(numerator.detach().cpu()))
            setattr(self, f"{attr_prefix}_recall_total", getattr(self, f"{attr_prefix}_recall_total") + float(denominator.detach().cpu()))

        for class_index in range(1, len(PIECE_CHANNELS) + 1):
            pred_mask = predicted_square_state == class_index
            target_mask = target_square_state == class_index
            self.piece_tp[class_index - 1] += float((pred_mask & target_mask).sum().item())
            self.piece_fp[class_index - 1] += float((pred_mask & ~target_mask).sum().item())
            self.piece_fn[class_index - 1] += float((~pred_mask & target_mask).sum().item())

        occupied_targets = target_square_state[occupied_mask]
        occupied_predictions = predicted_square_state[occupied_mask]
        for target_class, predicted_class in zip(occupied_targets.reshape(-1), occupied_predictions.reshape(-1)):
            row_index = int(target_class.item()) - 1
            column_index = int(predicted_class.item())
            self.non_empty_confusion[row_index, column_index] += 1

    def compute(self) -> dict[str, object]:
        """누적 통계로부터 최종 상태 충실도 지표를 계산한다."""
        legality_precision = self.legality_tp / max(self.legality_tp + self.legality_fp, 1.0)
        legality_recall = self.legality_tp / max(self.legality_tp + self.legality_fn, 1.0)
        legality_f1 = (
            2 * legality_precision * legality_recall / max(legality_precision + legality_recall, 1e-8)
        )
        piece_precision = self.piece_tp / torch.clamp(self.piece_tp + self.piece_fp, min=1.0)
        piece_recall = self.piece_tp / torch.clamp(self.piece_tp + self.piece_fn, min=1.0)
        piece_f1 = 2 * piece_precision * piece_recall / torch.clamp(piece_precision + piece_recall, min=1e-8)
        piece_macro_f1 = float(piece_f1.mean().item())
        return {
            "square_state_accuracy": self.square_correct / max(self.square_total, 1.0),
            "occupied_square_accuracy": self.occupied_correct / max(self.occupied_total, 1.0),
            "piece_macro_f1": piece_macro_f1,
            "piece_count_exact_match": self.piece_count_exact_match_sum / max(self.sample_count, 1.0),
            "king_count_validity": self.king_count_valid_sum / max(self.sample_count, 1.0),
            "side_to_move_accuracy": self.side_correct / max(self.side_total, 1.0),
            "castling_right_accuracy": self.castling_correct / max(self.castling_total, 1.0),
            "en_passant_accuracy": self.en_passant_correct / max(self.en_passant_total, 1.0),
            "in_check_accuracy": self.in_check_correct / max(self.in_check_total, 1.0),
            "legality_precision": legality_precision,
            "legality_recall": legality_recall,
            "legality_f1": legality_f1,
            "legality_average_precision": self.legality_ap_weighted_sum / max(self.legality_positive_total, 1.0),
            "legality_recall_at_16": self.legality_recall_at_16_sum / max(self.legality_recall_at_16_count, 1.0),
            "promotion_legality_recall": self.promotion_recall_tp / max(self.promotion_recall_total, 1.0),
            "castling_legality_recall": self.castling_recall_tp / max(self.castling_recall_total, 1.0),
            "en_passant_legality_recall": self.en_passant_recall_tp / max(self.en_passant_recall_total, 1.0),
            "check_evasion_legality_recall": self.check_evasion_recall_tp / max(self.check_evasion_recall_total, 1.0),
            "non_empty_confusion_labels": ["empty"] + list(PIECE_CHANNELS),
            "non_empty_confusion_matrix": self.non_empty_confusion.tolist(),
        }


def compute_state_fidelity_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, object]:
    """단일 배치의 상태 충실도 지표를 계산한다."""
    accumulator = StateFidelityAccumulator()
    accumulator.update(outputs, batch)
    return accumulator.compute()
