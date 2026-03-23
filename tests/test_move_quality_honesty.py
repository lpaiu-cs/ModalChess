import math

import pytest
import torch

from modalchess.eval.metrics_move_quality import compute_move_quality_metrics


def _build_batch() -> dict[str, object]:
    return {
        "legal_moves_factorized": [[(0, 1, 0), (1, 2, 0)]],
        "target_move_uci": ["a1b1"],
        "target_is_promotion": torch.tensor([False]),
        "target_is_castling": torch.tensor([False]),
        "target_is_en_passant": torch.tensor([False]),
        "subset_check_evasion": torch.tensor([False]),
        "concept_tags": [[]],
        "position_ids": ["p1"],
        "fens": ["8/8/8/8/8/8/8/8 w - - 0 1"],
    }


def _base_outputs() -> dict[str, torch.Tensor]:
    src_logits = torch.full((1, 64), -10.0)
    dst_logits = torch.full((1, 64), -10.0)
    promo_logits = torch.full((1, 5), -10.0)
    src_logits[0, 0] = 2.0
    src_logits[0, 1] = 1.0
    src_logits[0, 2] = 3.0
    dst_logits[0, 1] = 2.0
    dst_logits[0, 2] = 1.0
    dst_logits[0, 3] = 3.0
    promo_logits[0, 0] = 0.0
    return {
        "src_logits": src_logits,
        "dst_logits": dst_logits,
        "promo_logits": promo_logits,
    }


def test_move_quality_honesty_diagnostics_match_exact_raw_distribution() -> None:
    outputs = _base_outputs()
    metrics = compute_move_quality_metrics(outputs, _build_batch(), topk=(1, 3, 5))

    raw_scores = (
        outputs["src_logits"][0][:, None, None]
        + outputs["dst_logits"][0][None, :, None]
        + outputs["promo_logits"][0][None, None, :]
    ).reshape(-1)
    raw_probabilities = torch.softmax(raw_scores, dim=0)
    legal_indices = torch.tensor([0 * 64 * 5 + 1 * 5 + 0, 1 * 64 * 5 + 2 * 5 + 0], dtype=torch.long)
    target_index = 0 * 64 * 5 + 1 * 5 + 0
    expected_legal_mass = float(raw_probabilities.index_select(0, legal_indices).sum())
    expected_illegal_mass = 1.0 - expected_legal_mass
    expected_raw_target_nll = float(
        torch.nn.functional.cross_entropy(
            raw_scores.unsqueeze(0),
            torch.tensor([target_index], dtype=torch.long),
        )
    )

    assert metrics["honesty_diagnostics_status"] == "exact"
    assert metrics["illegal_top_1_rate"] == pytest.approx(1.0)
    assert metrics["raw_top_1_is_legal_rate"] == pytest.approx(0.0)
    assert metrics["legal_mass"] == pytest.approx(expected_legal_mass)
    assert metrics["illegal_mass"] == pytest.approx(expected_illegal_mass)
    assert metrics["raw_target_move_nll"] == pytest.approx(expected_raw_target_nll)
    assert metrics["raw_top_1"] == pytest.approx(0.0)
    assert metrics["raw_top_3"] == pytest.approx(0.0)
    assert metrics["raw_top_5"] == pytest.approx(1.0)
    assert math.isclose(metrics["legal_mass"] + metrics["illegal_mass"], 1.0, rel_tol=0.0, abs_tol=1e-6)


def test_move_quality_honesty_diagnostics_support_pair_scorer_on_and_off() -> None:
    batch = _build_batch()
    without_pair = compute_move_quality_metrics(_base_outputs(), batch, topk=(1, 3, 5))

    with_pair_outputs = _base_outputs()
    pair_logits = torch.zeros((1, 64, 64))
    pair_logits[0, 0, 1] = 5.0
    with_pair_outputs["pair_logits"] = pair_logits
    with_pair = compute_move_quality_metrics(with_pair_outputs, batch, topk=(1, 3, 5))

    for metrics in (without_pair, with_pair):
        for key in (
            "illegal_top_1_rate",
            "legal_mass",
            "illegal_mass",
            "raw_top_1_is_legal_rate",
            "raw_target_move_nll",
            "raw_top_1",
            "raw_top_3",
            "raw_top_5",
        ):
            assert key in metrics

    assert without_pair["raw_top_1_is_legal_rate"] == pytest.approx(0.0)
    assert without_pair["illegal_top_1_rate"] == pytest.approx(1.0)
    assert with_pair["raw_top_1_is_legal_rate"] == pytest.approx(1.0)
    assert with_pair["illegal_top_1_rate"] == pytest.approx(0.0)
