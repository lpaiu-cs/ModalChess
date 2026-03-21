from __future__ import annotations

import json
from pathlib import Path

import chess
import torch

from modalchess.data.probe_corpora import build_probe_corpora
from modalchess.data.probe_reports import generate_probe_corpora_report, write_probe_reports
from modalchess.data.probe_targets import build_probe_targets
from modalchess.eval.embedding_export import EmbeddingExportConfig, export_embeddings_for_checkpoint
from modalchess.eval.language_readiness import run_language_readiness_probes
from modalchess.train.train_spatial_baseline import build_model_from_config


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_probe_corpora_targets_and_reports(tmp_path: Path) -> None:
    mate_input = tmp_path / "language_mate.jsonl"
    puzzle_input = tmp_path / "puzzle_eval.jsonl"
    _write_jsonl(
        mate_input,
        [
            {
                "position_id": "mate_row_1",
                "game_id": "mate_game_1",
                "fen": chess.STARTING_FEN,
                "candidate_moves": ["e2e4", "d2d4"],
                "strategy_text": "Deliver check and attack the king.",
                "tactic_text": "A discovered attack can follow.",
                "preferred_move": "MoveA:e2e4",
                "source": "mate",
            },
            {
                "position_id": "mate_row_2",
                "game_id": "mate_game_2",
                "fen": "8/8/8/8/8/8/4K3/7k w - - 0 1",
                "candidate_moves": ["e2e3"],
                "strategy_text": "Create an open file for the rook.",
                "tactic_text": "No direct tactic.",
                "preferred_move": "MoveA:e2e3",
                "source": "mate",
            },
        ],
    )
    _write_jsonl(
        puzzle_input,
        [
            {
                "position_id": "puzzle_row_1",
                "game_id": "puzzle_game_1",
                "fen": chess.STARTING_FEN,
                "target_move_uci": "e2e4",
                "concept_tags": ["opening", "long"],
                "source": "lichess_puzzle",
                "history_fens": [chess.STARTING_FEN],
            },
            {
                "position_id": "puzzle_row_2",
                "game_id": "puzzle_game_2",
                "fen": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                "target_move_uci": "e1g1",
                "concept_tags": ["castling", "short"],
                "source": "lichess_puzzle",
                "history_fens": ["r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"],
            },
        ],
    )

    output_root = tmp_path / "language_probe_v1"
    build_probe_corpora(
        mate_path=mate_input,
        puzzle_path=puzzle_input,
        output_root=output_root,
    )
    build_probe_targets(
        input_root=output_root,
        output_root=output_root,
    )
    write_probe_reports(
        input_root=output_root,
        output_dir=output_root / "reports",
    )

    mate_rows = []
    for split_name in ("train", "val", "test"):
        mate_rows.extend(
            json.loads(line)
            for line in (output_root / f"mate_{split_name}.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
    assert {row["source_row_id"] for row in mate_rows} == {"mate_row_1", "mate_row_2"}
    assert all(row["probe_id"].startswith("probe_") for row in mate_rows)

    mate_target_rows = []
    for split_name in ("train", "val", "test"):
        mate_target_rows.extend(
            json.loads(line)
            for line in (output_root / f"mate_targets_{split_name}.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
    labels = {label for row in mate_target_rows for label in row["target_labels"]}
    assert "check" in labels
    assert "discovered_attack" in labels
    assert "open_file" in labels

    puzzle_target_rows = []
    for split_name in ("train", "val", "test"):
        puzzle_target_rows.extend(
            json.loads(line)
            for line in (output_root / f"puzzle_targets_{split_name}.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
    castling_row = next(row for row in puzzle_target_rows if row["target_move_uci"] == "e1g1")
    assert castling_row["castling_flag"] is True
    report = generate_probe_corpora_report(input_root=output_root)
    assert "mate" in report["label_counts"]
    assert "puzzle" in report["label_counts"]


def test_pt_embedding_export(tmp_path: Path) -> None:
    probe_path = tmp_path / "mate_train.jsonl"
    fen = chess.STARTING_FEN
    _write_jsonl(
        probe_path,
        [
            {
                "probe_id": "probe_a",
                "source": "mate",
                "split": "train",
                "fen": fen,
                "history_fens": [fen],
                "target_move_uci": None,
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
        dataset_paths={"mate_train": probe_path},
        output_dir=output_dir,
        config=EmbeddingExportConfig(output_format="pt"),
    )
    payload = torch.load(output_dir / "mate_train_embeddings.pt", map_location="cpu")
    assert payload["probe_id"] == ["probe_a"]
    assert payload["board_pooled"].shape == (1, 16)
    assert payload["context_pooled"].shape == (1, 16)


def test_run_language_readiness_probes(tmp_path: Path) -> None:
    target_root = tmp_path / "targets"
    embedding_root = tmp_path / "embeddings"
    output_dir = tmp_path / "probe_outputs"
    target_root.mkdir(parents=True, exist_ok=True)

    def write_target_split(family: str, split_name: str, rows: list[dict[str, object]]) -> None:
        _write_jsonl(target_root / f"{family}_targets_{split_name}.jsonl", rows)

    def embedding_payload(prefix: str, split_name: str, features: list[list[float]]) -> dict[str, object]:
        probe_ids = [f"{prefix}_{split_name}_{index}" for index in range(len(features))]
        return {
            "probe_id": probe_ids,
            "position_id": probe_ids,
            "source": [prefix] * len(features),
            "split": [split_name] * len(features),
            "fen": [chess.STARTING_FEN] * len(features),
            "target_move_uci": [None] * len(features),
            "board_pooled": torch.tensor(features, dtype=torch.float32),
            "context_pooled": torch.tensor(features, dtype=torch.float32),
            "checkpoint_path": "dummy",
            "seed": 11,
            "git_hash": "dummy",
            "model_parameter_count": 1,
        }

    for family in ("mate", "puzzle"):
        train_features = [[3.0, 0.0], [2.5, 0.1], [-3.0, 0.0], [-2.5, -0.1]]
        val_features = [[2.0, 0.0], [-2.0, 0.0]]
        test_features = [[2.2, 0.0], [-2.2, 0.0]]
        train_probe_ids = [f"{family}_train_{index}" for index in range(len(train_features))]
        val_probe_ids = [f"{family}_val_{index}" for index in range(len(val_features))]
        test_probe_ids = [f"{family}_test_{index}" for index in range(len(test_features))]
        write_target_split(
            family,
            "train",
            [
                {"probe_id": train_probe_ids[0], "target_labels": ["attack"]},
                {"probe_id": train_probe_ids[1], "target_labels": ["attack"]},
                {"probe_id": train_probe_ids[2], "target_labels": []},
                {"probe_id": train_probe_ids[3], "target_labels": []},
            ],
        )
        write_target_split(
            family,
            "val",
            [
                {"probe_id": val_probe_ids[0], "target_labels": ["attack"]},
                {"probe_id": val_probe_ids[1], "target_labels": []},
            ],
        )
        write_target_split(
            family,
            "test",
            [
                {"probe_id": test_probe_ids[0], "target_labels": ["attack"]},
                {"probe_id": test_probe_ids[1], "target_labels": []},
            ],
        )
        for backbone in ("g1", "g3"):
            seed_dir = embedding_root / backbone / "seed11"
            seed_dir.mkdir(parents=True, exist_ok=True)
            train_payload = embedding_payload(family, "train", train_features)
            train_payload["probe_id"] = train_probe_ids
            val_payload = embedding_payload(family, "val", val_features)
            val_payload["probe_id"] = val_probe_ids
            test_payload = embedding_payload(family, "test", test_features)
            test_payload["probe_id"] = test_probe_ids
            torch.save(train_payload, seed_dir / f"{family}_train_embeddings.pt")
            torch.save(val_payload, seed_dir / f"{family}_val_embeddings.pt")
            torch.save(test_payload, seed_dir / f"{family}_test_embeddings.pt")

    result = run_language_readiness_probes(
        embedding_root=embedding_root,
        target_root=target_root,
        output_dir=output_dir,
        backbone_seed=11,
        mate_min_train_positive=1,
        puzzle_min_train_positive=1,
    )
    assert result["week7_state_candidate"] in {
        "READY_FOR_TINY_FROZEN_ALIGNMENT",
        "STILL_EVAL_ONLY_BUT_STABLE",
        "DATA_EXPANSION_FIRST",
    }
    assert (output_dir / "probe_results.json").exists()
    assert (output_dir / "probe_results_aggregate.csv").exists()
    assert (output_dir / "readiness_probe_summary.md").exists()
