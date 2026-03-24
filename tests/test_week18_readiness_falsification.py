from __future__ import annotations

import pytest

from modalchess.eval.readiness_falsification import (
    PoolThresholds,
    build_family_deranged_permutation,
    classify_shared_pool,
    selection_epochs_from_epoch_metrics,
)


def test_classify_shared_pool_uses_requested_buckets() -> None:
    thresholds = PoolThresholds(primary_min_test_rows=500, strong_primary_min_test_rows=1000, exploratory_max_test_rows=199)
    assert classify_shared_pool(150, thresholds) == "exploratory_only"
    assert classify_shared_pool(437, thresholds) == "secondary_shared"
    assert classify_shared_pool(500, thresholds) == "primary_shared"
    assert classify_shared_pool(1000, thresholds) == "strong_primary"


def test_family_deranged_permutation_changes_every_family_label() -> None:
    labels = ["a", "a", "b", "b", "c", "c"]
    permutation = build_family_deranged_permutation(labels, seed=7)
    assert permutation.tolist() != list(range(len(labels)))
    for index, mapped in enumerate(permutation.tolist()):
        assert mapped != index
        assert labels[mapped] != labels[index]


def test_family_deranged_permutation_raises_when_majority_family_blocks_derangement() -> None:
    labels = ["a", "a", "a", "b", "c"]
    with pytest.raises(ValueError):
        build_family_deranged_permutation(labels, seed=3)


def test_selection_epochs_reports_policy_grounding_collapse() -> None:
    epoch_metrics = [
        {
            "epoch": 1,
            "val": {
                "target_move_nll": 2.6,
                "occupied_square_accuracy": 0.99,
                "piece_macro_f1": 0.97,
                "legality_average_precision": 0.10,
            },
        },
        {
            "epoch": 2,
            "val": {
                "target_move_nll": 2.4,
                "occupied_square_accuracy": 0.995,
                "piece_macro_f1": 0.99,
                "legality_average_precision": 0.40,
            },
        },
    ]
    selection = selection_epochs_from_epoch_metrics(epoch_metrics)
    assert selection["policy_best_epoch"] == 2
    assert selection["grounding_best_epoch"] == 2
    assert selection["epochs_match"] is True
