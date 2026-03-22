from __future__ import annotations

import json
from pathlib import Path

from modalchess.data.comment_informativeness import (
    CommentInformativenessConfig,
    compute_comment_informativeness,
    generate_comment_informativeness_audit,
)
from modalchess.data.informative_annotated_sidecar import (
    InformativeAnnotatedSidecarConfig,
    build_informative_annotated_sidecar,
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


def _base_row(sidecar_id: str, split: str, comment_text: str) -> dict[str, object]:
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
        "original_comment_text": comment_text,
        "comment_source": "comment",
        "split": split,
        "comment_template": comment_text.lower(),
    }


def test_informativeness_score_prefers_explanatory_chess_comment() -> None:
    config = CommentInformativenessConfig()
    low_row = _base_row("low", "train", "good move")
    high_row = _base_row(
        "high",
        "train",
        "The rook comes to e1 because it pins the knight and threatens mate on e8.",
    )
    low_score = compute_comment_informativeness(low_row, config=config)
    high_score = compute_comment_informativeness(high_row, config=config)
    assert high_score["informativeness_score"] > low_score["informativeness_score"]
    assert high_score["informativeness_bucket"] in {"medium", "high"}


def test_informativeness_audit_reports_bucket_counts(tmp_path: Path) -> None:
    input_root = tmp_path / "annotated_sidecar_v2_clean"
    _write_jsonl(
        input_root / "train.jsonl",
        [
            _base_row("s1", "train", "good move"),
            _base_row("s2", "train", "The knight jumps to f7 because it forks king and rook."),
        ],
    )
    _write_jsonl(input_root / "val.jsonl", [])
    _write_jsonl(input_root / "test.jsonl", [])

    report = generate_comment_informativeness_audit(input_root=input_root)
    assert "train" in report["bucket_counts_by_split"]
    assert report["total_rows"] == 2


def test_build_informative_sidecar_filters_low_scoring_rows(tmp_path: Path) -> None:
    input_root = tmp_path / "annotated_sidecar_v2_clean"
    _write_jsonl(
        input_root / "train.jsonl",
        [
            _base_row("low", "train", "good move"),
            _base_row("medium", "train", "The rook reaches the open file and attacks the queen."),
            _base_row("high", "train", "The knight jumps to f7 because it forks king and rook and wins the exchange."),
        ],
    )
    _write_jsonl(
        input_root / "val.jsonl",
        [_base_row("val_high", "val", "Checks on h7 and threatens mate on g8.")],
    )
    _write_jsonl(
        input_root / "test.jsonl",
        [_base_row("test_high", "test", "The passed pawn promotes after the king is cut off.")],
    )

    result = build_informative_annotated_sidecar(
        input_root=input_root,
        output_root=tmp_path / "annotated_sidecar_v3_informative",
        config=InformativeAnnotatedSidecarConfig(
            primary_variant="medium_high_only",
            informativeness_config=CommentInformativenessConfig(
                medium_threshold=0.40,
                high_threshold=0.60,
            ),
        ),
    )

    train_rows = _read_jsonl(tmp_path / "annotated_sidecar_v3_informative" / "train.jsonl")
    kept_ids = {str(row["sidecar_id"]) for row in train_rows}
    assert "low" not in kept_ids
    assert "medium" in kept_ids
    assert "high" in kept_ids
    diff = json.loads(Path(result["diff_json"]).read_text(encoding="utf-8"))
    assert diff["after_total_rows"] < diff["before_total_rows"]
