from __future__ import annotations

import json
from pathlib import Path

import chess
import torch
import yaml

from modalchess.data.annotated_pgn_sidecar import (
    AnnotatedPgnSidecarConfig,
    build_annotated_pgn_sidecar,
    write_annotated_sidecar_report,
)
from modalchess.data.preprocessing_common import StableSplitConfig
from modalchess.data.probe_corpora import ProbeCorpusConfig, build_probe_corpora
from modalchess.data.probe_reports import compare_probe_split_roots
from modalchess.eval.raw_text_retrieval import run_raw_text_retrieval_probes


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


def _embedding_payload(probe_ids: list[str], split_name: str, features: list[list[float]]) -> dict[str, object]:
    return {
        "probe_id": probe_ids,
        "position_id": probe_ids,
        "source": ["synthetic"] * len(features),
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


def test_probe_corpora_manifest_emits_split_strategy_fields(tmp_path: Path) -> None:
    mate_path = tmp_path / "mate.jsonl"
    puzzle_path = tmp_path / "puzzle.jsonl"
    _write_jsonl(
        mate_path,
        [
            {
                "source_row_id": "mate_1",
                "fen": chess.STARTING_FEN,
                "strategy_text": "Develop pieces.",
                "tactic_text": "Keep the king safe.",
                "metadata": {"game_id": "shared_game"},
            },
            {
                "source_row_id": "mate_2",
                "fen": chess.STARTING_FEN,
                "strategy_text": "Control the center.",
                "tactic_text": "Look for tactics.",
                "metadata": {"game_id": "shared_game"},
            },
            {
                "source_row_id": "mate_3",
                "fen": chess.STARTING_FEN,
                "strategy_text": "Fallback row.",
                "tactic_text": "Unique game id.",
                "metadata": {"game_id": "unique_game"},
            },
        ],
    )
    _write_jsonl(
        puzzle_path,
        [
            {
                "source_row_id": "puzzle_1",
                "fen": chess.STARTING_FEN,
                "target_move_uci": "e2e4",
                "theme_tags": ["fork"],
            }
        ],
    )
    output_root = tmp_path / "probe"
    build_probe_corpora(
        mate_path=mate_path,
        puzzle_path=puzzle_path,
        output_root=output_root,
        config=ProbeCorpusConfig(
            split_config=StableSplitConfig(salt="week9_test_probe"),
            prefer_game_id_group_split=True,
            min_game_id_group_size=2,
        ),
    )
    manifest = yaml.safe_load((output_root / "manifests" / "probe_manifest.yaml").read_text(encoding="utf-8"))
    strategy = manifest["split_strategy_by_source"]["mate"]
    assert strategy["split_key_type"] == "game_id"
    assert strategy["repeated_group_count"] == 1
    assert strategy["max_group_size"] == 2
    assert strategy["fallback_reason"] is None


def test_build_annotated_pgn_sidecar_and_report(tmp_path: Path) -> None:
    input_root = tmp_path / "annotated"
    annotated_file = input_root / "annotated_pgn-data.jsonl-00000-of-00001"
    _write_jsonl(
        annotated_file,
        [
            {
                "pipeline_key": "annotated/test_game",
                "metadata": {},
                "text": (
                    '[Event "Test"]\n'
                    '[Site "https://example.com/game"]\n'
                    '[Date "2024.01.01"]\n'
                    '[Round "?"]\n'
                    '[White "Alpha"]\n'
                    '[Black "Beta"]\n'
                    '[Result "*"]\n'
                    '[Variant "Standard"]\n'
                    '\n'
                    '1. e4 {Good move.} e5 $1 2. Nf3 {Develops a knight.} *\n'
                ),
            }
        ],
    )
    compare_root = tmp_path / "language_probe_v4"
    for split_name in ("train", "val", "test"):
        _write_jsonl(compare_root / f"aux_board_anchored_{split_name}.jsonl", [])

    output_root = tmp_path / "annotated_sidecar_v1"
    build_result = build_annotated_pgn_sidecar(
        input_root=input_root,
        output_root=output_root,
        config=AnnotatedPgnSidecarConfig(
            split_config=StableSplitConfig(salt="week9_sidecar_test"),
            include_history_fens=True,
        ),
    )
    assert sum(build_result["split_counts"].values()) == 3
    rows = []
    for split_name in ("train", "val", "test"):
        rows.extend(_read_jsonl(output_root / f"{split_name}.jsonl"))
    assert {str(row["target_move_uci"]) for row in rows} == {"e2e4", "e7e5", "g1f3"}
    e4_row = next(row for row in rows if row["target_move_uci"] == "e2e4")
    assert e4_row["fen"] == chess.STARTING_FEN
    assert e4_row["comment_text"] == "Good move."
    assert e4_row["history_fens"][-1] == e4_row["fen"]
    e5_row = next(row for row in rows if row["target_move_uci"] == "e7e5")
    assert e5_row["comment_text"] == ""
    assert e5_row["nag_codes"] == [1]

    report_result = write_annotated_sidecar_report(
        input_root=output_root,
        output_dir=output_root / "reports",
        compare_aux_root=compare_root,
    )
    report = json.loads(Path(report_result["report_json"]).read_text(encoding="utf-8"))
    assert report["rows_with_valid_target_move_uci"] == 3
    assert report["rows_with_valid_next_fen"] == 3
    assert report["week8_aux_comparison"]["move_conditioned_text_row_delta"] == 2


def test_compare_probe_split_roots_ignores_fallback_reason_wording(tmp_path: Path) -> None:
    previous_root = tmp_path / "previous"
    current_root = tmp_path / "current"
    for root in (previous_root, current_root):
        (root / "manifests").mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "val", "test"):
            _write_jsonl(root / f"mate_{split_name}.jsonl", [])
            _write_jsonl(root / f"puzzle_{split_name}.jsonl", [])
    _write_jsonl(
        previous_root / "mate_train.jsonl",
        [{"source_row_id": "mate_1", "split": "train"}],
    )
    _write_jsonl(
        current_root / "mate_val.jsonl",
        [{"source_row_id": "mate_1", "split": "val"}],
    )
    previous_manifest = {
        "split_strategy_by_source": {
            "mate": {
                "split_key_type": "source_row_id",
                "repeated_group_count": 0,
                "max_group_size": 1,
                "candidate_game_id_rows": 1,
                "fallback_reason": "no_repeated_game_id_groups",
            },
            "puzzle": {
                "split_key_type": "source_row_id",
                "repeated_group_count": 0,
                "max_group_size": 1,
                "candidate_game_id_rows": 0,
                "fallback_reason": "missing_game_id",
            },
        }
    }
    current_manifest = {
        "split_strategy_by_source": {
            "mate": {
                "split_key_type": "source_row_id",
                "repeated_group_count": 0,
                "max_group_size": 1,
                "candidate_game_id_rows": 1,
                "fallback_reason": "no_repeated_game_id_groups_meeting_threshold",
            },
            "puzzle": {
                "split_key_type": "source_row_id",
                "repeated_group_count": 0,
                "max_group_size": 1,
                "candidate_game_id_rows": 0,
                "fallback_reason": "missing_game_id",
            },
        }
    }
    (previous_root / "manifests" / "probe_manifest.yaml").write_text(
        yaml.safe_dump(previous_manifest, sort_keys=False),
        encoding="utf-8",
    )
    (current_root / "manifests" / "probe_manifest.yaml").write_text(
        yaml.safe_dump(current_manifest, sort_keys=False),
        encoding="utf-8",
    )
    diff = compare_probe_split_roots(previous_root=previous_root, current_root=current_root)
    assert "fell back to source_row_id" in diff["potential_leakage_reduction_note"]


def test_comment_retrieval_family_runs_on_annotated_sidecar(tmp_path: Path) -> None:
    corpus_root = tmp_path / "annotated_sidecar_v1"
    embedding_root = tmp_path / "embedding_exports"
    for split_name, row_count in (("train", 4), ("val", 2), ("test", 2)):
        probe_ids = [f"annotated_{split_name}_{index}" for index in range(row_count)]
        rows = []
        features = []
        for index, probe_id in enumerate(probe_ids):
            attacking = index < max(1, row_count // 2)
            rows.append(
                {
                    "probe_id": probe_id,
                    "comment_text": "Attack the king with forcing moves." if attacking else "Improve piece activity quietly.",
                }
            )
            features.append([3.0, 0.0] if attacking else [-3.0, 0.0])
        _write_jsonl(corpus_root / f"annotated_sidecar_{split_name}.jsonl", rows)
        for backbone in ("g1", "g3"):
            for seed in (11, 17, 23):
                seed_dir = embedding_root / backbone / f"seed{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                payload = _embedding_payload(probe_ids, split_name, features)
                payload["seed"] = seed
                torch.save(payload, seed_dir / f"annotated_sidecar_{split_name}_embeddings.pt")

    result = run_raw_text_retrieval_probes(
        embedding_root=embedding_root,
        corpus_root=corpus_root,
        output_dir=tmp_path / "comment_retrieval",
        backbone_seeds=[11, 17, 23],
        mate_min_df=1,
        puzzle_min_df=1,
        max_vocab_size=32,
        families=["annotated_sidecar"],
        output_prefix="comment_retrieval",
    )
    assert result["aggregate"]
    assert result["aggregate"][0]["family"] == "annotated_sidecar"
    assert Path(result["summary_path"]).name == "comment_retrieval_summary.md"
