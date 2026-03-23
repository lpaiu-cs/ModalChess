from __future__ import annotations

import json
from pathlib import Path

from modalchess.data.balanced_multisource_sidecar import (
    BalancedMultisourceConfig,
    build_balanced_multisource_sidecar,
)
from modalchess.data.comment_source_style import (
    CommentSourceStyleAuditConfig,
    generate_comment_source_style_audit,
)
from modalchess.data.preprocessing_common import write_jsonl


def _make_row(index: int, *, split: str, source: str, source_family: str, comment_text: str, comment_source: str) -> dict[str, object]:
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
        "comment_text": comment_text,
        "original_comment_text": comment_text,
        "comment_source": comment_source,
        "source": source,
        "source_family": source_family,
        "split": split,
        "record_semantics": "annotated_pgn_comment" if source == "waterhorse_annotated_pgn" else "pairwise_explanation",
    }


def _write_rows(root: Path, rows: list[dict[str, object]]) -> None:
    for split_name in ("train", "val", "test"):
        write_jsonl(root / f"{split_name}.jsonl", [row for row in rows if row["split"] == split_name])


def test_source_style_audit_identifies_markup_heavy_family(tmp_path: Path) -> None:
    input_root = tmp_path / "annotated_sidecar_v4_multisource"
    rows: list[dict[str, object]] = []
    for split_name in ("train", "val", "test"):
        rows.extend(
            _make_row(
                index,
                split=split_name,
                source="waterhorse_annotated_pgn",
                source_family="gameknot.com",
                comment_text="Human explanation about king safety.",
                comment_source="comment",
            )
            for index in range(3)
        )
        rows.extend(
            _make_row(
                index,
                split=split_name,
                source="waterhorse_annotated_pgn",
                source_family="markup.family",
                comment_text="[%eval 0.1] [%clk 0:10:00]",
                comment_source="comment",
            )
            for index in range(3)
        )
    _write_rows(input_root, rows)

    report = generate_comment_source_style_audit(
        input_root=input_root,
        config=CommentSourceStyleAuditConfig(min_group_rows=1),
    )
    assert report["most_markup_heavy_families"][0]["group_name"] == "markup.family"
    assert report["most_markup_heavy_families"][0]["markup_heavy_rate"] > 0.5


def test_build_balanced_sidecar_caps_dominant_family_and_preserves_original_text(tmp_path: Path) -> None:
    input_root = tmp_path / "annotated_sidecar_v4_multisource"
    rows: list[dict[str, object]] = []
    for split_name, count in (("train", 5), ("val", 2), ("test", 2)):
        rows.extend(
            _make_row(
                index,
                split=split_name,
                source="waterhorse_annotated_pgn",
                source_family="gameknot.com",
                comment_text="[%eval 0.0] Strong human note.",
                comment_source="comment",
            )
            for index in range(count)
        )
        rows.extend(
            _make_row(
                index,
                split=split_name,
                source="mate_both_zip",
                source_family="mate_both_pairwise",
                comment_text="Pairwise explanation text.",
                comment_source="mate_both_pairwise_explanation",
            )
            for index in range(count)
        )
    _write_rows(input_root, rows)

    output_root = tmp_path / "annotated_sidecar_v5_balanced"
    result = build_balanced_multisource_sidecar(
        input_root=input_root,
        output_root=output_root,
        config=BalancedMultisourceConfig(
            family_caps={"train": 2, "val": 1, "test": 1},
            source_type_caps={"train": 3, "val": 2, "test": 2},
            salt="week17_test",
        ),
    )
    balanced_rows = [
        json.loads(line)
        for line in (output_root / "variants" / "family_balanced" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert sum(int(row["source_family"] == "gameknot.com") for row in balanced_rows) == 2
    normalized_rows = [
        json.loads(line)
        for line in (output_root / "variants" / "family_balanced_plus_style_normalized" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    normalized_gameknot_row = next(row for row in normalized_rows if row["source_family"] == "gameknot.com")
    assert normalized_gameknot_row["original_comment_text"].startswith("[%eval")
    assert normalized_gameknot_row["style_normalization_mode"] == "strip_pgn_markup"
    assert Path(result["manifest_path"]).exists()


def test_build_balanced_sidecar_source_type_variant_caps_by_source_type(tmp_path: Path) -> None:
    input_root = tmp_path / "annotated_sidecar_v4_multisource"
    rows: list[dict[str, object]] = []
    for split_name in ("train", "val", "test"):
        rows.extend(
            _make_row(
                index,
                split=split_name,
                source="waterhorse_annotated_pgn",
                source_family="family_a",
                comment_text="Human comment.",
                comment_source="comment",
            )
            for index in range(4)
        )
        rows.extend(
            _make_row(
                index,
                split=split_name,
                source="mate_testset_zip",
                source_family="family_b",
                comment_text="Pairwise explanation.",
                comment_source="mate_testset_pairwise_explanation",
            )
            for index in range(4)
        )
    _write_rows(input_root, rows)

    output_root = tmp_path / "annotated_sidecar_v5_balanced"
    build_balanced_multisource_sidecar(
        input_root=input_root,
        output_root=output_root,
        config=BalancedMultisourceConfig(
            family_caps={"train": 10, "val": 10, "test": 10},
            source_type_caps={"train": 2, "val": 1, "test": 1},
            salt="week17_test",
        ),
    )
    source_type_rows = [
        json.loads(line)
        for line in (output_root / "variants" / "source_type_balanced" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    waterhorse_count = sum(int(row["source"] == "waterhorse_annotated_pgn") for row in source_type_rows)
    mate_count = sum(int(row["source"] == "mate_testset_zip") for row in source_type_rows)
    assert waterhorse_count == 2
    assert mate_count == 2
