from __future__ import annotations

import json
from pathlib import Path

from modalchess.data.clean_annotated_sidecar import (
    CleanAnnotatedSidecarConfig,
    build_clean_annotated_sidecar,
)
from modalchess.data.comment_boilerplate_audit import (
    CommentBoilerplateConfig,
    analyze_comment_text,
    generate_comment_boilerplate_audit,
)


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


def _base_row(sidecar_id: str, split: str, comment_text: str, comment_source: str = "comment") -> dict[str, object]:
    return {
        "sidecar_id": sidecar_id,
        "probe_id": sidecar_id,
        "source": "waterhorse_annotated_pgn",
        "source_file": "sample.jsonl",
        "source_row_id": f"row_{sidecar_id}",
        "game_id": f"game_{split}",
        "position_id": f"pos_{sidecar_id}",
        "ply_index": 1,
        "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
        "target_move_uci": "a1a2",
        "next_fen": "8/8/8/8/8/8/K7/7k b - - 1 1",
        "comment_text": comment_text,
        "nag_codes": [],
        "history_fens": ["8/8/8/8/8/8/8/K6k w - - 0 1"],
        "split": split,
        "comment_source": comment_source,
        "metadata": {"headers": {}},
    }


def test_boilerplate_classifier_marks_engine_markup_and_templates() -> None:
    config = CommentBoilerplateConfig(repeated_template_min_count=2)
    markup = analyze_comment_text("[%eval 0.0]", template_count=5, config=config)
    assert "markup_only" in markup["categories"]
    assert "pgn_markup_heavy" in markup["categories"]

    result = analyze_comment_text("1-0 White wins.", template_count=5, config=config)
    assert "result_comment" in result["categories"]

    engine = analyze_comment_text("Inaccuracy. a4 was best.", template_count=5, config=config)
    assert "engine_template" in engine["categories"]


def test_boilerplate_audit_reports_category_counts(tmp_path: Path) -> None:
    input_root = tmp_path / "annotated_sidecar_v1"
    _write_jsonl(
        input_root / "train.jsonl",
        [
            _base_row("s1", "train", "[%eval 0.0]"),
            _base_row("s2", "train", "Inaccuracy. a4 was best."),
            _base_row("s3", "train", "Good attacking move on the kingside."),
        ],
    )
    _write_jsonl(input_root / "val.jsonl", [])
    _write_jsonl(input_root / "test.jsonl", [])

    report = generate_comment_boilerplate_audit(
        input_root=input_root,
        config=CommentBoilerplateConfig(repeated_template_min_count=1),
    )
    assert report["counts_by_boilerplate_category"]["pgn_markup_heavy"] == 1
    assert report["counts_by_boilerplate_category"]["engine_template"] == 1


def test_build_clean_annotated_sidecar_keeps_clean_comments_and_preserves_originals(tmp_path: Path) -> None:
    input_root = tmp_path / "annotated_sidecar_v1"
    _write_jsonl(
        input_root / "train.jsonl",
        [
            _base_row("drop_eval", "train", "[%eval 0.0]"),
            _base_row("drop_engine", "train", "Inaccuracy. a4 was best."),
            _base_row("keep_a", "train", "This wins the exchange with a simple tactic. [%clk 0:10:00]"),
            _base_row("cap_same_source_1", "train", "Good attacking move on the kingside."),
            _base_row("cap_same_source_2", "train", "Good attacking move on the kingside."),
            _base_row("keep_other_source", "train", "Good attacking move on the kingside.", comment_source="comment+nag"),
        ],
    )
    _write_jsonl(
        input_root / "val.jsonl",
        [_base_row("val_keep", "val", "Centralizes the rook for the endgame.")],
    )
    _write_jsonl(
        input_root / "test.jsonl",
        [_base_row("test_keep", "test", "Threatens mate on the back rank.")],
    )

    result = build_clean_annotated_sidecar(
        input_root=input_root,
        output_root=tmp_path / "annotated_sidecar_v2_clean",
        config=CleanAnnotatedSidecarConfig(
            primary_variant="keep_comment_source_balance",
            boilerplate_config=CommentBoilerplateConfig(repeated_template_min_count=2),
            template_cap_per_cluster=1,
            template_cap_per_source_cluster=1,
            salt="week13_test_clean",
        ),
    )

    train_rows = _read_jsonl(tmp_path / "annotated_sidecar_v2_clean" / "train.jsonl")
    kept_ids = {str(row["sidecar_id"]) for row in train_rows}
    assert "drop_eval" not in kept_ids
    assert "drop_engine" not in kept_ids
    assert "keep_a" in kept_ids
    assert len({"cap_same_source_1", "cap_same_source_2"} & kept_ids) == 1
    assert "keep_other_source" in kept_ids

    keep_a_row = next(row for row in train_rows if row["sidecar_id"] == "keep_a")
    assert keep_a_row["original_comment_text"] == "This wins the exchange with a simple tactic. [%clk 0:10:00]"
    assert keep_a_row["comment_text"] == "This wins the exchange with a simple tactic."

    diff = json.loads(Path(result["diff_json"]).read_text(encoding="utf-8"))
    assert diff["after_total_rows"] < diff["before_total_rows"]
