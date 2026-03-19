import json
from pathlib import Path

import pytest

from modalchess.data.dataset_builder import DatasetBuildConfig, build_dataset


def test_jsonl_dataset_build_and_game_split(tmp_path: Path) -> None:
    dataset_path = tmp_path / "positions.jsonl"
    records = [
        {
            "position_id": "g1_p1",
            "game_id": "g1",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "target_move_uci": "e2e4",
        },
        {
            "position_id": "g1_p2",
            "game_id": "g1",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "target_move_uci": "e7e5",
        },
        {
            "position_id": "g2_p1",
            "game_id": "g2",
            "fen": "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            "target_move_uci": "a7a8q",
        },
    ]
    with dataset_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    train_dataset = build_dataset(
        DatasetBuildConfig(
            source="jsonl",
            dataset_path=str(dataset_path),
            split="train",
            split_seed=1,
            train_ratio=0.5,
            val_ratio=0.0,
        )
    )
    test_dataset = build_dataset(
        DatasetBuildConfig(
            source="jsonl",
            dataset_path=str(dataset_path),
            split="test",
            split_seed=1,
            train_ratio=0.5,
            val_ratio=0.0,
        )
    )

    train_game_ids = {sample.game_id for sample in train_dataset.samples}
    test_game_ids = {sample.game_id for sample in test_dataset.samples}
    assert train_game_ids
    assert test_game_ids
    assert train_game_ids.isdisjoint(test_game_ids)


def test_jsonl_dataset_validates_illegal_target_upfront(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid_positions.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "position_id": "bad_1",
                    "game_id": "g1",
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "target_move_uci": "e7e5",
                }
            )
            + "\n"
        )

    with pytest.raises(ValueError, match="target_move_uci"):
        build_dataset(
            DatasetBuildConfig(
                source="jsonl",
                dataset_path=str(dataset_path),
                split="all",
            )
        )


def test_jsonl_dataset_validates_history_alignment_upfront(tmp_path: Path) -> None:
    dataset_path = tmp_path / "bad_history.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "position_id": "bad_history",
                    "game_id": "g1",
                    "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                    "history_fens": [
                        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    ],
                    "target_move_uci": "e7e5",
                }
            )
            + "\n"
        )

    with pytest.raises(ValueError, match="history_fens"):
        build_dataset(
            DatasetBuildConfig(
                source="jsonl",
                dataset_path=str(dataset_path),
                split="all",
            )
        )


def test_jsonl_dataset_requires_game_id_for_group_split(tmp_path: Path) -> None:
    dataset_path = tmp_path / "missing_game_id.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "position_id": "p1",
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "target_move_uci": "e2e4",
                }
            )
            + "\n"
        )

    with pytest.raises(ValueError, match="game_id"):
        build_dataset(
            DatasetBuildConfig(
                source="jsonl",
                dataset_path=str(dataset_path),
                split="train",
                train_ratio=1.0,
                val_ratio=0.0,
            )
        )

    dataset = build_dataset(
        DatasetBuildConfig(
            source="jsonl",
            dataset_path=str(dataset_path),
            split="train",
            train_ratio=1.0,
            val_ratio=0.0,
            allow_position_level_split=True,
        )
    )
    assert len(dataset.samples) == 1


def test_jsonl_dataset_validates_provided_legal_moves(tmp_path: Path) -> None:
    dataset_path = tmp_path / "bad_legal_moves.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "position_id": "bad_legal_moves",
                    "game_id": "g1",
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "legal_moves_uci": ["e2e4"],
                    "target_move_uci": "e2e4",
                }
            )
            + "\n"
        )

    with pytest.raises(ValueError, match="legal_moves_uci"):
        build_dataset(
            DatasetBuildConfig(
                source="jsonl",
                dataset_path=str(dataset_path),
                split="all",
            )
        )


def test_jsonl_dataset_requires_target_move_when_next_fen_is_present(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dangling_next_fen.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "position_id": "dangling_next",
                    "game_id": "g1",
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "next_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                }
            )
            + "\n"
        )

    with pytest.raises(ValueError, match="next_fen"):
        build_dataset(
            DatasetBuildConfig(
                source="jsonl",
                dataset_path=str(dataset_path),
                split="all",
            )
        )


def test_jsonl_dataset_supports_explicit_split_field(tmp_path: Path) -> None:
    dataset_path = tmp_path / "explicit_split.jsonl"
    records = [
        {
            "position_id": "train_1",
            "game_id": "g1",
            "split": "train",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "target_move_uci": "e2e4",
        },
        {
            "position_id": "val_1",
            "game_id": "g2",
            "split": "val",
            "fen": "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            "target_move_uci": "a7a8q",
        },
        {
            "position_id": "test_1",
            "game_id": "g3",
            "split": "test",
            "fen": "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
            "target_move_uci": "e1e2",
        },
    ]
    with dataset_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    val_dataset = build_dataset(
        DatasetBuildConfig(
            source="jsonl",
            dataset_path=str(dataset_path),
            split="val",
            split_field="split",
        )
    )

    assert len(val_dataset.samples) == 1
    assert val_dataset.samples[0].position_id == "val_1"
