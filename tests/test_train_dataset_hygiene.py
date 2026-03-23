import json
from pathlib import Path

from modalchess.train.train_spatial_baseline import run_training


def test_run_training_uses_train_split_when_base_jsonl_dataset_is_reused(tmp_path: Path) -> None:
    dataset_path = tmp_path / "explicit_split_train.jsonl"
    records = [
        {
            "position_id": "train_1",
            "game_id": "g1",
            "split": "train",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "target_move_uci": "e2e4",
        },
        {
            "position_id": "train_2",
            "game_id": "g2",
            "split": "train",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "target_move_uci": "e7e5",
        },
        {
            "position_id": "val_1",
            "game_id": "g3",
            "split": "val",
            "fen": "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            "target_move_uci": "a7a8q",
        },
        {
            "position_id": "test_1",
            "game_id": "g4",
            "split": "test",
            "fen": "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
            "target_move_uci": "e1e2",
        },
    ]
    with dataset_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    output_dir = tmp_path / "train_output"
    metrics = run_training(
        {
            "seed": 7,
            "output_dir": str(output_dir),
            "dataset": {
                "source": "jsonl",
                "dataset_path": str(dataset_path),
                "history_length": 1,
            },
            "model": {
                "history_length": 1,
                "input_channels": 18,
                "d_model": 32,
                "num_layers": 1,
                "num_heads": 4,
                "mlp_ratio": 2,
                "dropout": 0.0,
                "use_relation_bias": False,
                "legality_hidden_dim": 16,
                "use_pair_scorer": False,
                "meta_num_tokens": 2,
                "concept_vocab": [],
            },
            "train": {
                "batch_size": 1,
                "eval_batch_size": 1,
                "epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "grad_clip_norm": 1.0,
                "run_overfit": False,
            },
            "losses": {
                "policy": 1.0,
                "policy_axis_ce": 1.0,
                "policy_listwise": 1.0,
                "state_probe": 1.0,
                "legality": 0.0,
                "value": 0.0,
                "concept": 0.0,
            },
        }
    )

    assert metrics["train_dataset_size"] == 2
    assert metrics["val_dataset_size"] == 1
