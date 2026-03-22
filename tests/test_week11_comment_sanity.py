from __future__ import annotations

import torch

from modalchess.eval.comment_retrieval_sanity import compute_rank_diagnostics
from modalchess.eval.week10_artifact_verification import choose_verification_gate


def test_choose_verification_gate_prefers_missing_or_mismatch_failures() -> None:
    assert choose_verification_gate(missing_count=1, mismatch_count=0, suspicious_findings=[]) == "NOT_VERIFIED"
    assert choose_verification_gate(missing_count=0, mismatch_count=1, suspicious_findings=[]) == "NOT_VERIFIED"
    assert (
        choose_verification_gate(
            missing_count=0,
            mismatch_count=0,
            suspicious_findings=["board_to_text_r1_equals_r5_in_aggregate"],
        )
        == "VERIFIED_BUT_NEEDS_SANITY_AUDIT"
    )
    assert choose_verification_gate(missing_count=0, mismatch_count=0, suspicious_findings=[]) == "VERIFIED_GOOD"


def test_compute_rank_diagnostics_penalizes_ties_in_strict_mode() -> None:
    scores = torch.tensor(
        [
            [1.0, 1.0, 0.2],
            [0.1, 0.9, 0.9],
            [0.1, 0.2, 0.8],
        ],
        dtype=torch.float32,
    )
    diagnostics = compute_rank_diagnostics(scores)
    assert diagnostics["standard_recall_at_1"] > diagnostics["strict_recall_at_1"]
    assert diagnostics["queries_with_tied_competitors"] == 2
    assert diagnostics["rank_2_to_5_count"] == 0
