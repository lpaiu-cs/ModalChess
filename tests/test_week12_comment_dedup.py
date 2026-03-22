from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from modalchess.data.comment_duplicate_audit import (
    generate_comment_duplicate_audit,
    normalize_comment_text,
)
from modalchess.data.comment_retrieval_eval import CommentRetrievalEvalConfig
from modalchess.data.dedup_comment_eval import (
    DedupCommentEvalConfig,
    build_comment_retrieval_eval_regime_v2,
    build_dedup_comment_eval,
)
from modalchess.eval.raw_text_retrieval import _chunked_retrieval_metrics_with_ties


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_normalize_comment_text_supports_week12_views() -> None:
    text = "  Attack!!   $1  "
    assert normalize_comment_text(text, mode="raw") == "Attack!!   $1"
    assert normalize_comment_text(text, mode="lower_ws") == "attack!! $1"
    assert normalize_comment_text(text, mode="punct_light") == "attack 1"
    assert normalize_comment_text(text, mode="nag_stripped") == "attack"


def test_generate_comment_duplicate_audit_reports_cross_position_reuse(tmp_path: Path) -> None:
    input_root = tmp_path / "annotated"
    shared_comment = "Attack the king!"
    rows = {
        "train": [
            {
                "sidecar_id": "s1",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "target_move_uci": "a1a2",
                "comment_text": shared_comment,
                "comment_source": "comment",
            },
            {
                "sidecar_id": "s2",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "target_move_uci": "a1a2",
                "comment_text": shared_comment,
                "comment_source": "comment",
            },
        ],
        "val": [
            {
                "sidecar_id": "s3",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "target_move_uci": "a1b1",
                "comment_text": shared_comment,
                "comment_source": "comment+nag",
            }
        ],
        "test": [],
    }
    for split_name, split_rows in rows.items():
        _write_jsonl(input_root / f"{split_name}.jsonl", split_rows)

    report = generate_comment_duplicate_audit(input_root=input_root)
    assert report["exact_comment_text"]["duplicate_row_count"] == 2
    assert report["duplicate_comment_text_across_positions"]["cluster_count"] == 1
    assert report["normalization_views"]["nag_stripped"]["cluster_summary"]["cluster_count"] >= 1


def test_build_dedup_comment_eval_and_regime_v2_write_expected_outputs(tmp_path: Path) -> None:
    input_root = tmp_path / "annotated_sidecar_v1"
    sample_rows = {
        "train": [
            {
                "sidecar_id": "train_1",
                "probe_id": "train_1",
                "comment_text": "Attack the king!",
                "comment_source": "comment",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "target_move_uci": "a1a2",
                "split": "train",
            },
            {
                "sidecar_id": "train_2",
                "probe_id": "train_2",
                "comment_text": "attack the king!!",
                "comment_source": "comment",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "target_move_uci": "a1b1",
                "split": "train",
            },
            {
                "sidecar_id": "train_3",
                "probe_id": "train_3",
                "comment_text": "",
                "comment_source": "nag",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "target_move_uci": "a1b2",
                "split": "train",
            },
        ],
        "val": [
            {
                "sidecar_id": "val_1",
                "probe_id": "val_1",
                "comment_text": "Attack the king!",
                "comment_source": "comment",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "target_move_uci": "a1a2",
                "split": "val",
            }
        ],
        "test": [
            {
                "sidecar_id": "test_1",
                "probe_id": "test_1",
                "comment_text": "Attack the king!",
                "comment_source": "comment",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "target_move_uci": "a1a2",
                "split": "test",
            }
        ],
    }
    for split_name, split_rows in sample_rows.items():
        _write_jsonl(input_root / f"{split_name}.jsonl", split_rows)

    build_result = build_dedup_comment_eval(
        input_root=input_root,
        output_root=tmp_path / "annotated_sidecar_eval_v2",
        config=DedupCommentEvalConfig(
            primary_variant="normalized_comment_dedup",
            normalized_mode="punct_light",
            capped_duplicates_per_cluster=2,
            salt="week12_test_dedup",
        ),
    )
    primary_train_rows = _read_jsonl(tmp_path / "annotated_sidecar_eval_v2" / "train.jsonl")
    assert len(primary_train_rows) == 2
    assert all("dedup_cluster_id" in row for row in primary_train_rows)

    regime_result = build_comment_retrieval_eval_regime_v2(
        input_root=tmp_path / "annotated_sidecar_eval_v2",
        output_root=tmp_path / "comment_retrieval_v2",
        config=CommentRetrievalEvalConfig(
            train_limit=10,
            val_limit=5,
            test_limit=5,
            salt="week12_eval_regime",
            require_non_empty_comment=True,
            stratify_by="comment_source",
        ),
    )
    manifest = yaml.safe_load(Path(regime_result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["corpus_mode"] == "dedup_aware"
    assert manifest["dedup_mode"] == "normalized_comment_dedup"

    dedup_report = json.loads(Path(build_result["report_json"]).read_text(encoding="utf-8"))
    assert "normalized_comment_dedup" in dedup_report["variants"]


def test_chunked_retrieval_metrics_with_ties_supports_strict_mode() -> None:
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    keys = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    optimistic = _chunked_retrieval_metrics_with_ties(queries, keys, tie_mode="optimistic")
    strict = _chunked_retrieval_metrics_with_ties(queries, keys, tie_mode="strict")
    assert optimistic[0] > strict[0]
