"""합법 수 집합 위에서 계산하는 이동 품질 지표."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from modalchess.data.move_codec import uci_to_factorized
from modalchess.models.heads.policy_factorized import score_factorized_moves


def collect_move_prediction_rows(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, object],
    topk: Sequence[int] = (1, 3, 5),
) -> list[dict[str, object]]:
    """샘플별 합법 수 랭킹과 subset 태그를 수집한다."""
    rows: list[dict[str, object]] = []
    max_topk = max(topk) if topk else 1
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
        top_predictions = []
        for rank_index, move_tuple in enumerate(ranked_moves[:max_topk]):
            top_predictions.append(
                {
                    "rank": rank_index + 1,
                    "src_square": move_tuple[0],
                    "dst_square": move_tuple[1],
                    "promotion": move_tuple[2],
                    "score": float(scores[ranking[rank_index]].detach().cpu()),
                }
            )
        rows.append(
            {
                "batch_index": index,
                "position_id": batch["position_ids"][index],
                "fen": batch["fens"][index],
                "target_move_uci": target_move,
                "target_move": {
                    "src_square": target_tuple[0],
                    "dst_square": target_tuple[1],
                    "promotion": target_tuple[2],
                },
                "predicted_top_1": top_predictions[0],
                "top_predictions": top_predictions,
                "is_correct_top_1": ranked_moves[0] == target_tuple,
                "is_promotion": bool(batch["target_is_promotion"][index].item()),
                "is_castling": bool(batch["target_is_castling"][index].item()),
                "is_en_passant": bool(batch["target_is_en_passant"][index].item()),
                "is_check_evasion": bool(batch["subset_check_evasion"][index].item()),
            }
        )
    return rows


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
    rows = collect_move_prediction_rows(outputs, batch, topk=topk)
    for row in rows:
        index = row["batch_index"]
        legal_moves = batch["legal_moves_factorized"][index]
        target_tuple = (
            row["target_move"]["src_square"],
            row["target_move"]["dst_square"],
            row["target_move"]["promotion"],
        )
        sample_outputs = {
            "src_logits": outputs["src_logits"][index],
            "dst_logits": outputs["dst_logits"][index],
            "promo_logits": outputs["promo_logits"][index],
        }
        if "pair_logits" in outputs:
            sample_outputs["pair_logits"] = outputs["pair_logits"][index]
        scores = score_factorized_moves(sample_outputs, legal_moves)
        target_index = legal_moves.index(target_tuple)
        nll_values.append(
            F.cross_entropy(
                scores.unsqueeze(0),
                torch.tensor([target_index], dtype=torch.long, device=scores.device),
            )
        )
        total += 1
        for k in topk:
            if any(
                prediction["src_square"] == target_tuple[0]
                and prediction["dst_square"] == target_tuple[1]
                and prediction["promotion"] == target_tuple[2]
                for prediction in row["top_predictions"][:k]
            ):
                correct[k] += 1
        if row["is_promotion"]:
            subset_total["promotion"] += 1
            subset_correct["promotion"] += int(row["is_correct_top_1"])
        if row["is_castling"]:
            subset_total["castling"] += 1
            subset_correct["castling"] += int(row["is_correct_top_1"])
        if row["is_en_passant"]:
            subset_total["en_passant"] += 1
            subset_correct["en_passant"] += int(row["is_correct_top_1"])
        if row["is_check_evasion"]:
            subset_total["check_evasion"] += 1
            subset_correct["check_evasion"] += int(row["is_correct_top_1"])
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
