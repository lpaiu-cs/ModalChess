"""합법 수 집합 위에서 계산하는 이동 품질 지표."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from modalchess.data.move_codec import uci_to_factorized
from modalchess.models.heads.policy_factorized import score_factorized_moves


def compute_move_quality_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, object],
    topk: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """평가 시점 합법 수 필터링을 사용해 top-k 이동 정확도를 계산한다."""
    correct = {k: 0 for k in topk}
    total = 0
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
        total += 1
        for k in topk:
            if target_tuple in ranked_moves[:k]:
                correct[k] += 1
    metrics = {}
    for k in topk:
        metrics[f"top_{k}_move_accuracy"] = float(correct[k] / total) if total else 0.0
    return metrics
