from __future__ import annotations

import json
from pathlib import Path

import chess
import torch

from modalchess.data.language_alignment import build_language_alignment_index
from modalchess.data.rationale_sidecar import build_rationale_sidecars
from modalchess.data.language_sidecar_report import generate_language_sidecar_report
from modalchess.eval.embedding_export import export_embeddings_for_checkpoint
from modalchess.train.train_spatial_baseline import build_model_from_config


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _supervised_row(position_id: str, game_id: str, fen: str, target_move_uci: str) -> dict[str, object]:
    board = chess.Board(fen)
    next_board = board.copy(stack=False)
    next_board.push_uci(target_move_uci)
    return {
        "position_id": position_id,
        "game_id": game_id,
        "fen": fen,
        "history_fens": [fen],
        "target_move_uci": target_move_uci,
        "next_fen": next_board.fen(en_passant="fen"),
    }


def test_language_alignment_preserves_ambiguity_and_move_conditioning(tmp_path: Path) -> None:
    start_fen = chess.STARTING_FEN
    fen4_only_supervised = "8/8/8/8/8/8/4K3/7k w - - 0 1"
    fen4_only_sidecar = "8/8/8/8/8/8/4K3/7k w - - 12 34"
    supervised_root = tmp_path / "supervised"
    _write_jsonl(
        supervised_root / "train.jsonl",
        [
            _supervised_row("train_exact", "game_train_exact", start_fen, "e2e4"),
            _supervised_row("train_fen4", "game_train_fen4", fen4_only_supervised, "e2e3"),
        ],
    )
    _write_jsonl(
        supervised_root / "val.jsonl",
        [_supervised_row("val_exact", "game_val_exact", start_fen, "d2d4")],
    )
    _write_jsonl(
        supervised_root / "test.jsonl",
        [_supervised_row("test_exact", "game_test_exact", "4k3/8/8/8/8/8/4K3/8 w - - 0 1", "e2e3")],
    )

    mate_path = tmp_path / "language_mate.jsonl"
    _write_jsonl(
        mate_path,
        [
            {
                "position_id": "mate_ambiguous",
                "game_id": "mate_game_ambiguous",
                "source": "mate",
                "fen": start_fen,
                "candidate_moves": ["e2e4", "d2d4"],
                "strategy_text": "A: Occupy the center. || B: Control key squares.",
                "tactic_text": "A: Central pawn push. || B: Queen pawn setup.",
                "preferred_move": "MoveA:e2e4",
            },
            {
                "position_id": "mate_fen4",
                "game_id": "mate_game_fen4",
                "source": "mate",
                "fen": fen4_only_sidecar,
                "candidate_moves": ["e2e3"],
                "strategy_text": "A: Improve king activity.",
                "tactic_text": "A: Step forward safely.",
                "preferred_move": "MoveA:e2e3",
            },
        ],
    )

    puzzle_path = tmp_path / "puzzle_eval.jsonl"
    _write_jsonl(
        puzzle_path,
        [
            {
                "position_id": "puzzle_disambiguated",
                "game_id": "puzzle_game",
                "fen": start_fen,
                "target_move_uci": "d2d4",
                "next_fen": chess.Board(start_fen).variation_san([chess.Move.from_uci("d2d4")]) if False else None,
                "concept_tags": ["opening", "center"],
                "source": "lichess_puzzle",
                "history_fens": [start_fen],
            }
        ],
    )

    output_root = tmp_path / "language_v1"
    build_language_alignment_index(
        supervised_train_path=supervised_root / "train.jsonl",
        supervised_val_path=supervised_root / "val.jsonl",
        supervised_test_path=supervised_root / "test.jsonl",
        mate_path=mate_path,
        puzzle_path=puzzle_path,
        output_root=output_root,
    )

    mate_val_rows = [json.loads(line) for line in (output_root / "mate_matched_val.jsonl").read_text(encoding="utf-8").splitlines()]
    mate_train_rows = [json.loads(line) for line in (output_root / "mate_matched_train.jsonl").read_text(encoding="utf-8").splitlines()]
    mate_unmatched_rows = [json.loads(line) for line in (output_root / "mate_unmatched.jsonl").read_text(encoding="utf-8").splitlines()]
    puzzle_val_rows = [json.loads(line) for line in (output_root / "puzzle_matched_val.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(mate_train_rows) == 1
    assert mate_train_rows[0]["alignment_type"] == "fen_4field"
    assert mate_train_rows[0]["matched_position_id"] == "train_fen4"
    assert mate_val_rows == []
    assert len(mate_unmatched_rows) == 1
    assert "ambigu" in mate_unmatched_rows[0]["notes"]
    assert len(puzzle_val_rows) == 1
    assert puzzle_val_rows[0]["alignment_type"] == "fen_exact_target_move"
    assert puzzle_val_rows[0]["matched_position_id"] == "val_exact"


def test_rationale_builder_and_sidecar_report(tmp_path: Path) -> None:
    alignment_root = tmp_path / "language_v1"
    castling_fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    _write_jsonl(
        alignment_root / "mate_matched_train.jsonl",
        [
            {
                "sidecar_id": "sidecar_train",
                "source": "mate",
                "source_row_id": "mate_train",
                "split": "train",
                "matched_supervised": True,
                "matched_position_id": "train_pos",
                "matched_game_id": "train_game",
                "fen": castling_fen,
                "target_move_uci": None,
                "matched_target_move_uci": "e1g1",
                "candidate_moves": ["e1g1"],
                "strategy_text": "A: Improve the king's safety. || B: Delay castling.",
                "tactic_text": "A: Castle to complete development. || B: Keep the king central.",
                "theme_tags": ["castling"],
                "alignment_type": "fen_exact",
                "alignment_confidence": 1.0,
                "notes": None,
                "preferred_move": "MoveA:e1g1",
                "history_fens": [castling_fen],
            }
        ],
    )
    _write_jsonl(
        alignment_root / "mate_matched_val.jsonl",
        [
            {
                "sidecar_id": "sidecar_val",
                "source": "mate",
                "source_row_id": "mate_val",
                "split": "val",
                "matched_supervised": True,
                "matched_position_id": "val_pos",
                "matched_game_id": "val_game",
                "fen": castling_fen,
                "target_move_uci": None,
                "matched_target_move_uci": "e1g1",
                "candidate_moves": ["e1g1"],
                "strategy_text": "A: Improve the king's safety.",
                "tactic_text": "A: Castle to complete development.",
                "theme_tags": ["castling"],
                "alignment_type": "fen_exact",
                "alignment_confidence": 1.0,
                "notes": None,
                "preferred_move": "MoveA:e1g1",
                "history_fens": [castling_fen],
            }
        ],
    )
    _write_jsonl(
        alignment_root / "mate_matched_test.jsonl",
        [
            {
                "sidecar_id": "sidecar_test",
                "source": "mate",
                "source_row_id": "mate_test",
                "split": "test",
                "matched_supervised": True,
                "matched_position_id": "test_pos",
                "matched_game_id": "test_game",
                "fen": castling_fen,
                "target_move_uci": None,
                "matched_target_move_uci": "e1g1",
                "candidate_moves": ["e1g1"],
                "strategy_text": "A: Improve the king's safety.",
                "tactic_text": "A: Castle to complete development.",
                "theme_tags": ["castling"],
                "alignment_type": "fen_exact",
                "alignment_confidence": 1.0,
                "notes": None,
                "preferred_move": "MoveA:e1g1",
                "history_fens": [castling_fen],
            }
        ],
    )
    _write_jsonl(alignment_root / "mate_unmatched.jsonl", [])
    _write_jsonl(alignment_root / "puzzle_matched_val.jsonl", [])
    _write_jsonl(alignment_root / "puzzle_matched_test.jsonl", [])

    build_rationale_sidecars(
        input_root=alignment_root,
        output_root=alignment_root,
    )
    rationale_val_rows = [
        json.loads(line)
        for line in (alignment_root / "rationale_val.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rationale_val_rows) == 1
    rationale_row = rationale_val_rows[0]
    assert rationale_row["castling_flag"] is True
    assert rationale_row["target_move_uci"] == "e1g1"
    assert "Castle to complete development" in rationale_row["rationale_short"]
    assert "e1" in rationale_row["focus_squares"]
    assert "g1" in rationale_row["focus_squares"]

    report = generate_language_sidecar_report(input_root=alignment_root)
    assert report["leakage_check"]["passes"] is True
    assert report["matched_counts_by_split"] == {"train": 1, "val": 1, "test": 1}
    assert report["readiness"]["status"] == "partially_ready"


def test_embedding_export_for_language_rows(tmp_path: Path) -> None:
    rationale_path = tmp_path / "rationale_val.jsonl"
    fen = chess.STARTING_FEN
    _write_jsonl(
        rationale_path,
        [
            {
                "position_id": "val_pos",
                "split": "val",
                "source": "mate",
                "fen": fen,
                "history_fens": [fen],
                "target_move_uci": "e2e4",
                "sidecar_id": "sidecar_val",
                "alignment_type": "fen_exact",
            }
        ],
    )

    model_config = {
        "architecture": "spatial",
        "history_length": 1,
        "input_channels": 18,
        "d_model": 16,
        "num_layers": 1,
        "num_heads": 4,
        "mlp_ratio": 2,
        "dropout": 0.0,
        "use_relation_bias": False,
        "legality_hidden_dim": 16,
        "concept_vocab": [],
        "use_pair_scorer": True,
        "meta_num_tokens": 2,
    }
    model = build_model_from_config(model_config)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "resolved_model_config": model_config,
            "seed": 11,
            "git_hash": "testhash",
        },
        checkpoint_path,
    )

    output_dir = tmp_path / "embeddings"
    export_embeddings_for_checkpoint(
        checkpoint_path=checkpoint_path,
        dataset_paths={"rationale_val": rationale_path},
        output_dir=output_dir,
    )
    rows = [
        json.loads(line)
        for line in (output_dir / "rationale_val_embeddings.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["position_id"] == "val_pos"
    assert len(rows[0]["board_pooled"]) == 16
    assert len(rows[0]["context_pooled"]) == 16
