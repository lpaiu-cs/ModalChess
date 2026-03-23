from __future__ import annotations

import json
from pathlib import Path

import torch

from modalchess.data.preprocessing_common import write_jsonl
from modalchess.data.shared_source_holdout_eval import (
    SharedSourceHoldoutEvalConfig,
    build_shared_source_holdout_eval,
)
from modalchess.data.source_holdout_eval import HoldoutThresholds
from modalchess.eval.retrieval_comparison import _bootstrap_delta_ci, _strict_reciprocal_ranks
from modalchess.eval.winner_stability import build_winner_stability_report


def _write_split(root: Path, split_name: str, rows: list[dict[str, object]]) -> None:
    write_jsonl(root / f"{split_name}.jsonl", rows)


def test_shared_source_holdout_eval_uses_common_test_probe_ids(tmp_path: Path) -> None:
    variant_a = tmp_path / "variant_a"
    variant_b = tmp_path / "variant_b"
    for root in (variant_a, variant_b):
        root.mkdir(parents=True, exist_ok=True)

    common_train = [
        {"probe_id": f"train_{index}", "source": "srcA", "source_family": "famA", "comment_source": "comment", "comment_text": "alpha"}
        for index in range(3)
    ]
    common_val = [
        {"probe_id": f"val_{index}", "source": "srcA", "source_family": "famA", "comment_source": "comment", "comment_text": "beta"}
        for index in range(2)
    ]
    test_a = [
        {"probe_id": "shared_test", "source": "srcA", "source_family": "famA", "comment_source": "comment", "comment_text": "gamma"},
        {"probe_id": "only_a", "source": "srcA", "source_family": "famA", "comment_source": "comment", "comment_text": "delta"},
    ]
    test_b = [
        {"probe_id": "shared_test", "source": "srcA", "source_family": "famA", "comment_source": "comment", "comment_text": "gamma"},
        {"probe_id": "only_b", "source": "srcA", "source_family": "famA", "comment_source": "comment", "comment_text": "epsilon"},
    ]
    _write_split(variant_a, "train", common_train)
    _write_split(variant_a, "val", common_val)
    _write_split(variant_a, "test", test_a)
    _write_split(variant_b, "train", common_train)
    _write_split(variant_b, "val", common_val)
    _write_split(variant_b, "test", test_b)

    result = build_shared_source_holdout_eval(
        variant_roots={"a": variant_a, "b": variant_b},
        output_root=tmp_path / "holdout",
        config=SharedSourceHoldoutEvalConfig(
            coarse_thresholds=HoldoutThresholds(1, 1, 1, 1),
            family_thresholds=HoldoutThresholds(1, 1, 1, 1),
            min_shared_test_rows=1,
        ),
    )
    assert result["shared_regime_count"] >= 1
    shared_test_path = tmp_path / "holdout" / "variants" / "a" / "regimes" / "mixed_baseline" / "annotated_sidecar_test.jsonl"
    shared_rows = [json.loads(line) for line in shared_test_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["probe_id"] for row in shared_rows] == ["shared_test"]


def test_bootstrap_delta_ci_is_positive_for_positive_shift() -> None:
    deltas = _strict_reciprocal_ranks(
        query_vectors=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        key_vectors=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    ) - torch.tensor([0.25, 0.25])
    low, high = _bootstrap_delta_ci(deltas, samples=200, seed=7)
    assert low > 0.0
    assert high >= low


def test_winner_stability_report_counts_backbone_wins(tmp_path: Path) -> None:
    week17_results = {
        "comparison": [
            {
                "variant": "family_balanced",
                "best_board_backbone": "g3",
                "best_board_pool": "context_pooled",
                "best_board_probe_model": "linear",
                "best_strict_board_to_text_mrr_mean": 0.01,
                "best_text_backbone": "g1",
                "best_text_pool": "board_pooled",
                "best_text_probe_model": "mlp",
                "best_strict_text_to_board_mrr_mean": 0.02,
            }
        ]
    }
    week18_results = {
        "variants": {
            "family_balanced": {
                "aggregate": [
                    {
                        "regime_name": "mixed_baseline",
                        "backbone": "g3",
                        "pool": "context_pooled",
                        "probe_model": "linear",
                        "strict_board_to_text_mrr_mean": 0.02,
                        "strict_text_to_board_mrr_mean": 0.01,
                        "test_rows_mean": 100,
                    },
                    {
                        "regime_name": "mixed_baseline",
                        "backbone": "g1",
                        "pool": "board_pooled",
                        "probe_model": "mlp",
                        "strict_board_to_text_mrr_mean": 0.01,
                        "strict_text_to_board_mrr_mean": 0.03,
                        "test_rows_mean": 100,
                    },
                ],
                "results": [
                    {
                        "regime_name": "mixed_baseline",
                        "seed": 11,
                        "backbone": "g3",
                        "pool": "context_pooled",
                        "probe_model": "linear",
                        "strict_board_to_text_mrr": 0.02,
                        "strict_text_to_board_mrr": 0.01,
                    },
                    {
                        "regime_name": "mixed_baseline",
                        "seed": 11,
                        "backbone": "g1",
                        "pool": "board_pooled",
                        "probe_model": "mlp",
                        "strict_board_to_text_mrr": 0.01,
                        "strict_text_to_board_mrr": 0.03,
                    },
                ],
            }
        }
    }
    week17_path = tmp_path / "week17.json"
    week18_path = tmp_path / "week18.json"
    week17_path.write_text(json.dumps(week17_results), encoding="utf-8")
    week18_path.write_text(json.dumps(week18_results), encoding="utf-8")
    result = build_winner_stability_report(
        week17_results_path=week17_path,
        week18_holdout_path=week18_path,
        output_dir=tmp_path / "winner",
    )
    payload = result["payload"]
    assert payload["week18_regime_winner_counts"]["family_balanced"]["board_to_text"]["g3"] == 1
    assert payload["week18_regime_winner_counts"]["family_balanced"]["text_to_board"]["g1"] == 1
