from __future__ import annotations

from pathlib import Path

import torch

from modalchess.data.preprocessing_common import write_jsonl
from modalchess.data.source_holdout_eval import (
    HoldoutThresholds,
    SourceHoldoutEvalConfig,
    build_source_holdout_eval,
)
from modalchess.eval.source_holdout_retrieval import _filter_embedding_payload


def _make_row(index: int, *, split: str, source: str, source_family: str) -> dict[str, object]:
    sidecar_id = f"{source_family}_{split}_{index}"
    return {
        "sidecar_id": sidecar_id,
        "probe_id": sidecar_id,
        "game_id": f"g_{source_family}_{index}",
        "position_id": f"p_{source_family}_{index}",
        "ply_index": index,
        "fen": "8/8/8/8/8/8/8/4K3 w - - 0 1",
        "target_move_uci": "e1d2",
        "next_fen": "8/8/8/8/8/8/3K4/8 b - - 1 1",
        "comment_text": f"{source_family} explanation {index}",
        "original_comment_text": f"{source_family} explanation {index}",
        "comment_source": "comment" if source == "waterhorse_annotated_pgn" else f"{source_family}_explanation",
        "source": source,
        "source_family": source_family,
        "split": split,
    }


def _write_splits(root: Path, rows: list[dict[str, object]]) -> None:
    for split_name in ("train", "val", "test"):
        write_jsonl(root / f"{split_name}.jsonl", [row for row in rows if row["split"] == split_name])


def test_build_source_holdout_eval_prevents_leakage_and_excludes_tiny_sources(tmp_path: Path) -> None:
    input_root = tmp_path / "annotated_sidecar_eval_v5"
    rows: list[dict[str, object]] = []
    for split_name, count in (("train", 12), ("val", 4), ("test", 4)):
        rows.extend(
            _make_row(index, split=split_name, source="waterhorse_annotated_pgn", source_family="gameknot.com")
            for index in range(count)
        )
        rows.extend(
            _make_row(index, split=split_name, source="mate_both_zip", source_family="mate_both_pairwise")
            for index in range(count)
        )
    rows.append(_make_row(0, split="train", source="mate_tactic_zip", source_family="mate_tactic_pairwise"))
    _write_splits(input_root, rows)

    output_root = tmp_path / "annotated_sidecar_holdout_v1"
    result = build_source_holdout_eval(
        input_root=input_root,
        output_root=output_root,
        config=SourceHoldoutEvalConfig(
            coarse_thresholds=HoldoutThresholds(10, 10, 4, 4),
            family_thresholds=HoldoutThresholds(10, 10, 4, 4),
        ),
    )

    holdout_dir = output_root / "coarse_source_holdouts" / "waterhorse_annotated_pgn"
    train_rows = holdout_dir.joinpath("annotated_sidecar_train.jsonl").read_text(encoding="utf-8")
    val_rows = holdout_dir.joinpath("annotated_sidecar_val.jsonl").read_text(encoding="utf-8")
    test_rows = holdout_dir.joinpath("annotated_sidecar_test.jsonl").read_text(encoding="utf-8")
    assert "waterhorse_annotated_pgn" not in train_rows
    assert "waterhorse_annotated_pgn" not in val_rows
    assert "waterhorse_annotated_pgn" in test_rows
    assert result["regime_count"] >= 3

    report = output_root / "reports" / "holdout_design_report.json"
    payload = report.read_text(encoding="utf-8")
    assert "mate_tactic_zip" in payload
    assert "total<10" in payload


def test_build_source_holdout_eval_builds_source_type_regimes(tmp_path: Path) -> None:
    input_root = tmp_path / "annotated_sidecar_eval_v5"
    rows: list[dict[str, object]] = []
    for split_name, count in (("train", 6), ("val", 3), ("test", 3)):
        rows.extend(
            _make_row(index, split=split_name, source="waterhorse_annotated_pgn", source_family="lichess.org")
            for index in range(count)
        )
        rows.extend(
            _make_row(index, split=split_name, source="mate_testset_zip", source_family="mate_testset_pairwise")
            for index in range(count)
        )
    _write_splits(input_root, rows)

    output_root = tmp_path / "annotated_sidecar_holdout_v1"
    build_source_holdout_eval(
        input_root=input_root,
        output_root=output_root,
        config=SourceHoldoutEvalConfig(
            coarse_thresholds=HoldoutThresholds(1, 1, 1, 1),
            family_thresholds=HoldoutThresholds(1, 1, 1, 1),
        ),
    )

    waterhorse_text = (output_root / "source_types" / "waterhorse_only" / "annotated_sidecar_train.jsonl").read_text(
        encoding="utf-8"
    )
    mate_text = (output_root / "source_types" / "mate_only" / "annotated_sidecar_train.jsonl").read_text(
        encoding="utf-8"
    )
    assert "waterhorse_annotated_pgn" in waterhorse_text
    assert "mate_testset_zip" not in waterhorse_text
    assert "mate_testset_zip" in mate_text
    assert "waterhorse_annotated_pgn" not in mate_text


def test_filter_embedding_payload_preserves_requested_probe_order() -> None:
    payload = {
        "probe_id": ["a", "b", "c"],
        "position_id": ["pa", "pb", "pc"],
        "source": ["s1", "s2", "s3"],
        "split": ["train", "train", "train"],
        "fen": ["fa", "fb", "fc"],
        "target_move_uci": ["a1a2", "b1b2", "c1c2"],
        "board_pooled": torch.tensor([[1.0], [2.0], [3.0]]),
        "context_pooled": torch.tensor([[10.0], [20.0], [30.0]]),
        "checkpoint_path": "ckpt.pt",
        "seed": 11,
    }
    filtered = _filter_embedding_payload(payload, ["c", "a"])
    assert filtered["probe_id"] == ["c", "a"]
    assert filtered["board_pooled"].tolist() == [[3.0], [1.0]]
    assert filtered["context_pooled"].tolist() == [[30.0], [10.0]]
