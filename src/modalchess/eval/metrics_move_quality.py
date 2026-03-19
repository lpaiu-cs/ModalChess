"""합법 수 집합 위에서 계산하는 이동 품질 지표."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from modalchess.data.move_codec import uci_to_factorized
from modalchess.models.heads.policy_factorized import score_factorized_moves


def compute_move_quality_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, object],
    topk: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """평가 시점 합법 수 필터링을 사용해 top-k 이동 정확도를 계산한다."""
    correct = {k: 0 for k in topk}
    subset_correct = {
        "promotion": 0,
        "castling": 0,
        "en_passant": 0,
        "check_evasion": 0,
    }
    subset_total = {name: 0 for name in subset_correct}
    total = 0
    nll_values: list[torch.Tensor] = []
    for index, legal_moves in enumerate(batch["legal_moves_factorized"]):
        target_move = batch["target_move_uci"][index]
        if target_move is None or not legal_moves:
            continue
        sample_outputs = {
            "src_logits": outputs["src_logits"][index],
            "dst_logits": outputs["dst_logits"][index],
            "promo_logits": outputs["promo_logits"][index],
        }
        if "pair_logits" in outputs:
            sample_outputs["pair_logits"] = outputs["pair_logits"][index]
        scores = score_factorized_moves(sample_outputs, legal_moves)
        ranking = torch.argsort(scores, descending=True)
        ranked_moves = [legal_moves[i] for i in ranking.tolist()]
        factorized_target = uci_to_factorized(target_move)
        target_tuple = (
            factorized_target.src_square,
            factorized_target.dst_square,
            factorized_target.promotion,
        )
        target_index = legal_moves.index(target_tuple)
        nll_values.append(
            F.cross_entropy(
                scores.unsqueeze(0),
                torch.tensor([target_index], dtype=torch.long, device=scores.device),
            )
        )
        total += 1
        for k in topk:
            if target_tuple in ranked_moves[:k]:
                correct[k] += 1
        if bool(batch["target_is_promotion"][index].item()):
            subset_total["promotion"] += 1
            subset_correct["promotion"] += int(ranked_moves[0] == target_tuple)
        if bool(batch["target_is_castling"][index].item()):
            subset_total["castling"] += 1
            subset_correct["castling"] += int(ranked_moves[0] == target_tuple)
        if bool(batch["target_is_en_passant"][index].item()):
            subset_total["en_passant"] += 1
            subset_correct["en_passant"] += int(ranked_moves[0] == target_tuple)
        if bool(batch["subset_check_evasion"][index].item()):
            subset_total["check_evasion"] += 1
            subset_correct["check_evasion"] += int(ranked_moves[0] == target_tuple)
    metrics = {}
    for k in topk:
        metrics[f"top_{k}_move_accuracy"] = float(correct[k] / total) if total else 0.0
    metrics["target_move_nll"] = (
        float(torch.stack(nll_values).mean().detach().cpu()) if nll_values else 0.0
    )
    for subset_name, subset_count in subset_total.items():
        metrics[f"{subset_name}_top_1_move_accuracy"] = (
            float(subset_correct[subset_name] / subset_count) if subset_count else 0.0
        )
    return metrics
