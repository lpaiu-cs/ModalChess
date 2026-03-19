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
