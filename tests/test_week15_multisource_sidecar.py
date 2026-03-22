from __future__ import annotations

import json
from pathlib import Path

import pytest

from modalchess.data.comment_source_audit import derive_comment_source_family, generate_comment_source_family_audit
from modalchess.data.multisource_annotated_sidecar import (
    MultisourceAnnotatedSidecarConfig,
    _extract_mate_row,
    build_multisource_annotated_sidecar,
)
from modalchess.data.preprocessing_common import StableSplitConfig, write_jsonl


def test_derive_comment_source_family_prefers_url_domains() -> None:
    row = {
        "source": "waterhorse_annotated_pgn",
        "metadata": {
            "headers": {
                "Site": "https://lichess.org/study/abc",
                "Annotator": "https://gameknot.com/annotator/1",
            }
        },
    }
    assert derive_comment_source_family(row) == "lichess.org"


def test_extract_mate_row_parses_move_conditioned_explanation() -> None:
    row = {
        "instruction": "test",
        "input": (
            'The FEN of the given chess board is "8/8/8/8/8/8/8/4K3 w - - 0 1". '
            "Which move is better? MoveA:e1d2 Hold the king together. "
            "MoveB:e1f2 Walk into danger. "
        ),
        "output": "MoveA:e1d2",
    }
    parsed = _extract_mate_row(
        row,
        source_name="mate_strategy_zip",
        source_family="strategy",
        source_file="mate.zip::row.jsonl",
        line_index=0,
    )
    assert parsed is not None
    assert parsed["target_move_uci"] == "e1d2"
    assert parsed["comment_text"] == "Hold the king together."
    assert parsed["next_fen"]


def test_source_family_audit_reports_top_family(tmp_path: Path) -> None:
    rows = [
        {
            "sidecar_id": "s1",
            "fen": "8/8/8/8/8/8/8/4K3 w - - 0 1",
            "target_move_uci": "e1d2",
            "comment_text": "Because the king needs safety.",
            "comment_source": "comment",
            "source": "waterhorse_annotated_pgn",
            "source_file": "a.jsonl",
            "metadata": {"headers": {"Site": "https://lichess.org/study/abc"}},
            "split": "train",
        },
        {
            "sidecar_id": "s2",
            "fen": "8/8/8/8/8/8/8/4K3 w - - 0 1",
            "target_move_uci": "e1d2",
            "comment_text": "Because the king needs safety.",
            "comment_source": "comment",
            "source": "waterhorse_annotated_pgn",
            "source_file": "b.jsonl",
            "metadata": {"headers": {"Site": "https://gameknot.com/game/1"}},
            "split": "val",
        },
    ]
    for split_name in ("train", "val", "test"):
        write_jsonl(tmp_path / f"{split_name}.jsonl", [row for row in rows if row["split"] == split_name])
    report = generate_comment_source_family_audit(input_root=tmp_path)
    assert report["top_source_families"][0]["source_family"] in {"lichess.org", "gameknot.com"}
    assert report["source_family_diversity"]["family_count"] == 2


def test_build_multisource_sidecar_caps_dominant_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_root = tmp_path / "annotated_sidecar_v1"
    input_root.mkdir(parents=True, exist_ok=True)
    train_rows = [
        {
            "sidecar_id": f"w{i}",
            "fen": "8/8/8/8/8/8/8/4K3 w - - 0 1",
            "target_move_uci": "e1d2",
            "next_fen": "8/8/8/8/8/8/3K4/8 b - - 1 1",
            "comment_text": "A useful comment.",
            "comment_source": "comment",
            "source": "waterhorse_annotated_pgn",
            "source_file": "train.jsonl",
            "game_id": f"g{i}",
            "position_id": f"p{i}",
            "ply_index": i,
            "metadata": {"headers": {"Site": "https://gameknot.com/game/1"}},
            "split": "train",
        }
        for i in range(5)
    ]
    for split_name in ("train", "val", "test"):
        write_jsonl(input_root / f"{split_name}.jsonl", train_rows if split_name == "train" else [])

    monkeypatch.setattr(
        "modalchess.data.multisource_annotated_sidecar._mate_rows",
        lambda: [
            {
                "sidecar_id": "m1",
                "probe_id": "m1",
                "source": "mate_strategy_zip",
                "source_family": "mate_strategy_pairwise",
                "source_file": "mate.zip::a.jsonl",
                "source_row_id": "1",
                "game_id": "mate_game_1",
                "position_id": "mate_pos_1",
                "ply_index": 0,
                "fen": "8/8/8/8/8/8/8/4K3 w - - 0 1",
                "target_move_uci": "e1d2",
                "next_fen": "8/8/8/8/8/8/3K4/8 b - - 1 1",
                "comment_text": "Choose the safer king move.",
                "original_comment_text": "Choose the safer king move.",
                "comment_source": "mate_strategy_pairwise_explanation",
                "record_semantics": "pairwise_explanation",
                "metadata": {},
            }
        ],
    )

    output_root = tmp_path / "annotated_sidecar_v4_multisource"
    config = MultisourceAnnotatedSidecarConfig(
        split_config=StableSplitConfig(salt="week15_test"),
        source_family_caps={"train": 2, "val": 2, "test": 2},
        min_source_family_presence={"train": 1, "val": 0, "test": 0},
    )
    result = build_multisource_annotated_sidecar(
        waterhorse_input_root=input_root,
        output_root=output_root,
        config=config,
    )
    rows = [json.loads(line) for line in (output_root / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    families = [row["source_family"] for row in rows]
    assert families.count("gameknot.com") == 2
    assert "mate_strategy_pairwise" in families
    assert Path(result["manifest_path"]).exists()
