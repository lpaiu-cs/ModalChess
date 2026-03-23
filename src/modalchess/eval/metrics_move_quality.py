"""합법 수 집합 위에서 계산하는 이동 품질 지표."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from modalchess.data.move_codec import uci_to_factorized
from modalchess.models.heads.policy_factorized import build_raw_action_scores, score_factorized_moves


BASE_SUBSET_FLAGS = {
    "promotion": "is_promotion",
    "castling": "is_castling",
    "en_passant": "is_en_passant",
    "check_evasion": "is_check_evasion",
}

THEME_SUBSET_FLAGS = (
    "theme_promotion_related",
    "theme_check_related",
    "theme_castling_related",
    "theme_en_passant_related",
)


def _normalize_theme_tag(tag: str) -> str:
    return "".join(character for character in tag.lower() if character.isalnum())


def _theme_group_flags(concept_tags: Sequence[str] | None) -> dict[str, bool]:
    normalized = {_normalize_theme_tag(tag) for tag in concept_tags or []}
    return {
        "theme_promotion_related": any(
            "promotion" in tag or "underpromotion" in tag or "promote" in tag
            for tag in normalized
        ),
        "theme_check_related": any(
            "check" in tag or "mate" in tag or "evasion" in tag
            for tag in normalized
        ),
        "theme_castling_related": any(
            "castle" in tag or "castling" in tag
            for tag in normalized
        ),
        "theme_en_passant_related": any("enpassant" in tag for tag in normalized),
    }


def _prediction_matches_target(
    prediction: dict[str, object],
    target: dict[str, int],
) -> bool:
    return (
        prediction["src_square"] == target["src_square"]
        and prediction["dst_square"] == target["dst_square"]
        and prediction["promotion"] == target["promotion"]
    )


def _move_tuple_to_flat_index(move_tuple: tuple[int, int, int]) -> int:
    src_square, dst_square, promotion = move_tuple
    return src_square * 64 * 5 + dst_square * 5 + promotion


def _flat_index_to_move_tuple(flat_index: int) -> tuple[int, int, int]:
    src_square, remainder = divmod(flat_index, 64 * 5)
    dst_square, promotion = divmod(remainder, 5)
    return src_square, dst_square, promotion


def _prediction_payload_from_move_tuple(
    move_tuple: tuple[int, int, int],
    score: torch.Tensor,
    rank: int,
) -> dict[str, object]:
    return {
        "rank": rank,
        "src_square": move_tuple[0],
        "dst_square": move_tuple[1],
        "promotion": move_tuple[2],
        "score": float(score.detach().cpu()),
    }


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
        raw_scores = build_raw_action_scores(sample_outputs)
        flat_raw_scores = raw_scores.reshape(-1)
        raw_probabilities = torch.softmax(flat_raw_scores, dim=0)
        raw_ranking = torch.argsort(flat_raw_scores, descending=True)
        factorized_target = uci_to_factorized(target_move)
        target_tuple = (
            factorized_target.src_square,
            factorized_target.dst_square,
            factorized_target.promotion,
        )
        target_index = legal_moves.index(target_tuple)
        target_flat_index = _move_tuple_to_flat_index(target_tuple)
        legal_flat_indices = [_move_tuple_to_flat_index(move_tuple) for move_tuple in legal_moves]
        legal_flat_index_tensor = torch.tensor(
            legal_flat_indices,
            dtype=torch.long,
            device=flat_raw_scores.device,
        )
        legal_flat_index_set = set(legal_flat_indices)
        top_predictions = []
        for rank_index, move_tuple in enumerate(ranked_moves[:max_topk]):
            top_predictions.append(
                _prediction_payload_from_move_tuple(
                    move_tuple,
                    scores[ranking[rank_index]],
                    rank=rank_index + 1,
                )
            )
        raw_top_predictions = []
        for rank_index, flat_index in enumerate(raw_ranking[:max_topk].tolist()):
            move_tuple = _flat_index_to_move_tuple(flat_index)
            raw_top_predictions.append(
                _prediction_payload_from_move_tuple(
                    move_tuple,
                    flat_raw_scores[flat_index],
                    rank=rank_index + 1,
                )
            )
        target_payload = {
            "src_square": target_tuple[0],
            "dst_square": target_tuple[1],
            "promotion": target_tuple[2],
        }
        theme_flags = _theme_group_flags(batch.get("concept_tags", [])[index])
        topk_hits = {
            f"is_correct_top_{k}": any(
                _prediction_matches_target(prediction, target_payload)
                for prediction in top_predictions[:k]
            )
            for k in topk
        }
        raw_topk_hits = {
            f"is_correct_raw_top_{k}": any(
                _prediction_matches_target(prediction, target_payload)
                for prediction in raw_top_predictions[:k]
            )
            for k in topk
        }
        legal_mass = float(raw_probabilities.index_select(0, legal_flat_index_tensor).sum().detach().cpu())
        raw_top_1_is_legal = _move_tuple_to_flat_index(
            (
                int(raw_top_predictions[0]["src_square"]),
                int(raw_top_predictions[0]["dst_square"]),
                int(raw_top_predictions[0]["promotion"]),
            )
        ) in legal_flat_index_set
        rows.append(
            {
                "batch_index": index,
                "position_id": batch["position_ids"][index],
                "fen": batch["fens"][index],
                "target_move_uci": target_move,
                "target_move": target_payload,
                "predicted_top_1": top_predictions[0],
                "raw_predicted_top_1": raw_top_predictions[0],
                "top_predictions": top_predictions,
                "raw_top_predictions": raw_top_predictions,
                "is_correct_top_1": top_predictions[0]["src_square"] == target_tuple[0]
                and top_predictions[0]["dst_square"] == target_tuple[1]
                and top_predictions[0]["promotion"] == target_tuple[2],
                "target_move_nll": float(
                    F.cross_entropy(
                        scores.unsqueeze(0),
                        torch.tensor([target_index], dtype=torch.long, device=scores.device),
                    ).detach().cpu()
                ),
                "raw_target_move_nll": float(
                    F.cross_entropy(
                        flat_raw_scores.unsqueeze(0),
                        torch.tensor([target_flat_index], dtype=torch.long, device=flat_raw_scores.device),
                    ).detach().cpu()
                ),
                "raw_top_1_is_legal": raw_top_1_is_legal,
                "illegal_top_1": not raw_top_1_is_legal,
                "legal_mass": legal_mass,
                "illegal_mass": 1.0 - legal_mass,
                "is_promotion": bool(batch["target_is_promotion"][index].item()),
                "is_castling": bool(batch["target_is_castling"][index].item()),
                "is_en_passant": bool(batch["target_is_en_passant"][index].item()),
                "is_check_evasion": bool(batch["subset_check_evasion"][index].item()),
                "concept_tags": list(batch.get("concept_tags", [])[index]),
                **topk_hits,
                **raw_topk_hits,
                **theme_flags,
            }
        )
    return rows


def summarize_move_prediction_rows(
    rows: list[dict[str, object]],
    topk: Sequence[int] = (1, 3, 5),
) -> dict[str, object]:
    """샘플별 예측 행으로부터 최종 move-quality 지표를 계산한다."""
    correct = {k: 0 for k in topk}
    raw_correct = {k: 0 for k in topk}
    raw_field_names = {
        "raw_target_move_nll",
        "legal_mass",
        "illegal_mass",
        "raw_top_1_is_legal",
        "illegal_top_1",
        "raw_top_predictions",
    }
    raw_diagnostics_available = all(
        all(field_name in row for field_name in raw_field_names)
        for row in rows
    )
    subset_flag_names = dict(BASE_SUBSET_FLAGS)
    subset_flag_names.update(
        {
            subset_name: subset_name
            for subset_name in THEME_SUBSET_FLAGS
        }
    )
    subset_correct = {
        subset_name: {k: 0 for k in topk}
        for subset_name in subset_flag_names
    }
    subset_total = {name: 0 for name in subset_correct}
    subset_nll = {name: 0.0 for name in subset_correct}
    nll_total = 0.0
    raw_nll_total = 0.0
    legal_mass_total = 0.0
    illegal_mass_total = 0.0
    illegal_top_1_total = 0
    raw_top_1_is_legal_total = 0
    for row in rows:
        target = row["target_move"]
        nll_total += float(row["target_move_nll"])
        if raw_diagnostics_available:
            raw_nll_total += float(row["raw_target_move_nll"])
            legal_mass_total += float(row["legal_mass"])
            illegal_mass_total += float(row["illegal_mass"])
            illegal_top_1_total += int(bool(row["illegal_top_1"]))
            raw_top_1_is_legal_total += int(bool(row["raw_top_1_is_legal"]))
        for k in topk:
            if any(_prediction_matches_target(prediction, target) for prediction in row["top_predictions"][:k]):
                correct[k] += 1
            if raw_diagnostics_available and any(
                _prediction_matches_target(prediction, target)
                for prediction in row["raw_top_predictions"][:k]
            ):
                raw_correct[k] += 1
        for subset_name, flag_name in subset_flag_names.items():
            if row[flag_name]:
                subset_total[subset_name] += 1
                subset_nll[subset_name] += float(row["target_move_nll"])
                for k in topk:
                    subset_correct[subset_name][k] += int(row[f"is_correct_top_{k}"])
    total = len(rows)
    metrics: dict[str, object] = {"num_move_samples": total}
    for k in topk:
        accuracy = float(correct[k] / total) if total else 0.0
        metrics[f"top_{k}"] = accuracy
        metrics[f"top_{k}_move_accuracy"] = accuracy
    metrics["target_move_nll"] = nll_total / total if total else 0.0
    if raw_diagnostics_available:
        for k in topk:
            raw_accuracy = float(raw_correct[k] / total) if total else 0.0
            metrics[f"raw_top_{k}"] = raw_accuracy
            metrics[f"raw_top_{k}_move_accuracy"] = raw_accuracy
        metrics["raw_target_move_nll"] = raw_nll_total / total if total else 0.0
        metrics["legal_mass"] = legal_mass_total / total if total else 0.0
        metrics["illegal_mass"] = illegal_mass_total / total if total else 0.0
        metrics["raw_top_1_is_legal_rate"] = raw_top_1_is_legal_total / total if total else 0.0
        metrics["illegal_top_1_rate"] = illegal_top_1_total / total if total else 0.0
        metrics["honesty_diagnostics_status"] = "exact"
    else:
        metrics["honesty_diagnostics_status"] = "skipped_missing_raw_fields"
    subset_metrics: dict[str, dict[str, float | int]] = {}
    for subset_name, subset_count in subset_total.items():
        subset_payload: dict[str, float | int] = {"count": subset_count}
        for k in topk:
            subset_accuracy = (
                float(subset_correct[subset_name][k] / subset_count) if subset_count else 0.0
            )
            subset_payload[f"top_{k}"] = subset_accuracy
            subset_payload[f"top_{k}_move_accuracy"] = subset_accuracy
        subset_payload["target_move_nll"] = subset_nll[subset_name] / subset_count if subset_count else 0.0
        subset_metrics[subset_name] = subset_payload
        metrics[f"{subset_name}_count"] = subset_count
        for k in topk:
            metrics[f"{subset_name}_top_{k}"] = float(subset_payload[f"top_{k}"])
        metrics[f"{subset_name}_target_move_nll"] = float(subset_payload["target_move_nll"])
    metrics["subsets"] = subset_metrics
    return metrics


def compute_move_quality_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, object],
    topk: Sequence[int] = (1, 3, 5),
) -> dict[str, object]:
    """평가 시점 합법 수 필터링을 사용해 top-k 이동 정확도를 계산한다."""
    rows = collect_move_prediction_rows(outputs, batch, topk=topk)
    return summarize_move_prediction_rows(rows, topk=topk)
