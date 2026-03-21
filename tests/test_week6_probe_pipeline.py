from __future__ import annotations

import json
from pathlib import Path

import chess
import torch
import yaml

from modalchess.data.preprocessing_common import StableSplitConfig, assign_split_by_game_id
from modalchess.data.probe_corpora import ProbeCorpusConfig, build_probe_corpora
from modalchess.data.probe_reports import write_probe_reports
from modalchess.eval.language_readiness import run_language_readiness_probes
from modalchess.eval.language_retrieval import run_retrieval_readiness_probes


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _find_group_split_fixture() -> tuple[StableSplitConfig, str, list[str]]:
    config = StableSplitConfig(train_ratio=0.5, val_ratio=0.25, salt="week6_split_test")
    for index in range(1, 500):
        game_id = f"shared_game_{index}"
        source_row_ids = [f"mate_row_{index}_a", f"mate_row_{index}_b"]
        game_split = assign_split_by_game_id(game_id, config)
        source_splits = [assign_split_by_game_id(source_row_id, config) for source_row_id in source_row_ids]
        if any(source_split != game_split for source_split in source_splits):
            return config, game_id, source_row_ids
    raise AssertionError("game-aware split change fixture를 찾지 못했다.")


def _embedding_payload(
    probe_ids: list[str],
    split_name: str,
    features: list[list[float]],
) -> dict[str, object]:
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


def test_probe_split_hygiene_v2_and_diff_report(tmp_path: Path) -> None:
    split_config, shared_game_id, source_row_ids = _find_group_split_fixture()
    mate_input = tmp_path / "language_mate.jsonl"
    puzzle_input = tmp_path / "puzzle_eval.jsonl"
    fen = chess.STARTING_FEN
    _write_jsonl(
        mate_input,
        [
            {
                "position_id": source_row_ids[0],
                "game_id": shared_game_id,
                "fen": fen,
                "candidate_moves": ["e2e4"],
                "strategy_text": "Give check.",
                "tactic_text": "Check follows.",
                "source": "mate",
            },
            {
                "position_id": source_row_ids[1],
                "game_id": shared_game_id,
                "fen": "8/8/8/8/8/8/4K3/7k w - - 0 1",
                "candidate_moves": ["e2e3"],
                "strategy_text": "Create an open file.",
                "tactic_text": "No tactic.",
                "source": "mate",
            },
        ],
    )
    _write_jsonl(
        puzzle_input,
        [
            {
                "position_id": "puzzle_row_1",
                "game_id": "puzzle_unique_1",
                "fen": fen,
                "target_move_uci": "e2e4",
                "history_fens": [fen],
                "source": "lichess_puzzle",
            }
        ],
    )

    v1_root = tmp_path / "language_probe_v1"
    v2_root = tmp_path / "language_probe_v2"
    build_probe_corpora(
        mate_path=mate_input,
        puzzle_path=puzzle_input,
        output_root=v1_root,
        config=ProbeCorpusConfig(
            split_config=split_config,
            prefer_game_id_group_split=False,
        ),
    )
    build_probe_corpora(
        mate_path=mate_input,
        puzzle_path=puzzle_input,
        output_root=v2_root,
        config=ProbeCorpusConfig(
            split_config=split_config,
            prefer_game_id_group_split=True,
            min_game_id_group_size=2,
        ),
    )
    write_probe_reports(
        input_root=v2_root,
        output_dir=v2_root / "reports",
        compare_root=v1_root,
    )

    with (v2_root / "manifests" / "probe_manifest.yaml").open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    assert manifest["split_strategy_by_source"]["mate"]["split_key_type"] == "game_id"
    assert manifest["split_strategy_by_source"]["puzzle"]["split_key_type"] == "source_row_id"

    v2_mate_rows = [
        json.loads(line)
        for split_name in ("train", "val", "test")
        for line in (v2_root / f"mate_{split_name}.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len({row["split"] for row in v2_mate_rows}) == 1
    diff_payload = json.loads((v2_root / "reports" / "v1_vs_v2_split_diff.json").read_text(encoding="utf-8"))
    assert diff_payload["total_rows_with_split_change"] > 0


def test_multi_seed_readiness_and_retrieval_probes(tmp_path: Path) -> None:
    corpus_root = tmp_path / "language_probe_v2"
    embedding_root = tmp_path / "embeddings"
    readiness_output = tmp_path / "readiness_outputs"
    retrieval_output = tmp_path / "retrieval_outputs"

    families = ("mate", "puzzle")
    split_features = {
        "train": [[3.0, 0.0], [2.5, 0.1], [-3.0, 0.0], [-2.5, -0.1]],
        "val": [[2.0, 0.0], [-2.0, 0.0]],
        "test": [[2.2, 0.0], [-2.2, 0.0]],
    }

    for family in families:
        for split_name, features in split_features.items():
            probe_rows: list[dict[str, object]] = []
            target_rows: list[dict[str, object]] = []
            probe_ids = [f"{family}_{split_name}_{index}" for index in range(len(features))]
            for index, probe_id in enumerate(probe_ids):
                is_positive = index < max(1, len(features) // 2)
                if family == "mate":
                    probe_rows.append(
                        {
                            "probe_id": probe_id,
                            "source": "mate",
                            "source_row_id": probe_id,
                            "split": split_name,
                            "fen": chess.STARTING_FEN,
                            "strategy_text": "Give check to the king." if is_positive else "Improve piece placement.",
                            "tactic_text": "Check follows." if is_positive else "Quiet move.",
                        }
                    )
                    target_rows.append(
                        {
                            "probe_id": probe_id,
                            "source": "mate",
                            "split": split_name,
                            "fen": chess.STARTING_FEN,
                            "target_labels": ["check"] if is_positive else [],
                        }
                    )
                else:
                    probe_rows.append(
                        {
                            "probe_id": probe_id,
                            "source": "lichess_puzzle",
                            "source_row_id": probe_id,
                            "split": split_name,
                            "fen": chess.STARTING_FEN,
                            "target_move_uci": "e2e4",
                        }
                    )
                    target_rows.append(
                        {
                            "probe_id": probe_id,
                            "source": "lichess_puzzle",
                            "split": split_name,
                            "fen": chess.STARTING_FEN,
                            "target_move_uci": "e2e4",
                            "target_labels": ["fork"] if is_positive else [],
                            "promotion_flag": False,
                            "castling_flag": False,
                            "en_passant_flag": False,
                            "check_evasion_flag": False,
                        }
                    )
            _write_jsonl(corpus_root / f"{family}_{split_name}.jsonl", probe_rows)
            _write_jsonl(corpus_root / f"{family}_targets_{split_name}.jsonl", target_rows)

            for backbone in ("g1", "g3"):
                for seed in (11, 17, 23):
                    seed_dir = embedding_root / backbone / f"seed{seed}"
                    seed_dir.mkdir(parents=True, exist_ok=True)
                    payload = _embedding_payload(probe_ids, split_name, features)
                    payload["seed"] = seed
                    torch.save(payload, seed_dir / f"{family}_{split_name}_embeddings.pt")

    readiness_result = run_language_readiness_probes(
        embedding_root=embedding_root,
        target_root=corpus_root,
        output_dir=readiness_output,
        backbone_seeds=[11, 17, 23],
        mate_min_train_positive=1,
        puzzle_min_train_positive=1,
    )
    assert readiness_result["week7_state_candidate"] in {
        "READY_FOR_TINY_FROZEN_ALIGNMENT",
        "STILL_EVAL_ONLY_BUT_STABLE",
        "DATA_EXPANSION_FIRST",
    }
    assert readiness_result["aggregate"]
    assert all(row["seed_count"] == 3 for row in readiness_result["aggregate"])
    assert (readiness_output / "probe_results_aggregate.csv").exists()

    retrieval_result = run_retrieval_readiness_probes(
        embedding_root=embedding_root,
        corpus_root=corpus_root,
        target_root=corpus_root,
        output_dir=retrieval_output,
        backbone_seeds=[11, 17, 23],
        mate_min_train_positive=1,
        puzzle_min_train_positive=1,
    )
    assert retrieval_result["aggregate"]
    assert max(
        row["board_to_text_recall_at_1_mean"]
        for row in retrieval_result["aggregate"]
    ) >= 0.5
    assert (retrieval_output / "retrieval_summary.md").exists()
