import json
from pathlib import Path

import pytest

from modalchess.train.train_spatial_baseline import _compute_pareto_epochs, run_training


def _write_explicit_split_dataset(dataset_path: Path) -> None:
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
    ]
    dataset_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_grounding_best_checkpoint_selection_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "selection_dataset.jsonl"
    _write_explicit_split_dataset(dataset_path)

    eval_metrics_sequence = iter(
        [
            (
                {
                    "target_move_nll": 0.20,
                    "top_1": 0.5,
                    "top_3": 1.0,
                    "top_5": 1.0,
                    "occupied_square_accuracy": 0.50,
                    "piece_macro_f1": 0.40,
                    "legality_average_precision": 0.30,
                    "legality_f1": 0.25,
                },
                [],
            ),
            (
                {
                    "target_move_nll": 0.25,
                    "top_1": 0.4,
                    "top_3": 1.0,
                    "top_5": 1.0,
                    "occupied_square_accuracy": 0.90,
                    "piece_macro_f1": 0.80,
                    "legality_average_precision": 0.70,
                    "legality_f1": 0.50,
                },
                [],
            ),
        ]
    )

    monkeypatch.setattr(
        "modalchess.train.train_spatial_baseline.evaluate_model_on_dataloader",
        lambda *args, **kwargs: next(eval_metrics_sequence),
    )

    output_dir = tmp_path / "train_selection"
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
                "epochs": 2,
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

    assert metrics["best_epoch"] == 1
    assert metrics["policy_best_epoch"] == 1
    assert metrics["grounding_best_epoch"] == 2
    assert Path(metrics["best_checkpoint_path"]).exists()
    assert Path(metrics["best_policy_checkpoint_path"]).exists()
    assert Path(metrics["best_grounding_checkpoint_path"]).exists()
    assert Path(metrics["selection_summary_json"]).exists()
    assert Path(metrics["selection_summary_csv"]).exists()
    assert Path(metrics["pareto_epochs_json"]).exists()

    run_metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert run_metadata["policy_best_epoch"] == 1
    assert run_metadata["grounding_best_epoch"] == 2
    assert run_metadata["policy_selection_metric"] == "val.target_move_nll"
    assert "legality_average_precision" in run_metadata["grounding_selection_metric"]

    selection_summary = json.loads((output_dir / "selection_summary.json").read_text(encoding="utf-8"))
    assert selection_summary["policy_best_epoch"] == 1
    assert selection_summary["grounding_best_epoch"] == 2
    assert selection_summary["epochs"][0]["is_policy_best"] is True
    assert selection_summary["epochs"][1]["is_grounding_best"] is True


def test_pareto_summary_regression() -> None:
    pareto_epochs = _compute_pareto_epochs(
        [
            {
                "epoch": 1,
                "val": {
                    "target_move_nll": 0.20,
                    "occupied_square_accuracy": 0.50,
                    "piece_macro_f1": 0.45,
                    "legality_average_precision": 0.40,
                },
            },
            {
                "epoch": 2,
                "val": {
                    "target_move_nll": 0.25,
                    "occupied_square_accuracy": 0.90,
                    "piece_macro_f1": 0.85,
                    "legality_average_precision": 0.80,
                },
            },
            {
                "epoch": 3,
                "val": {
                    "target_move_nll": 0.30,
                    "occupied_square_accuracy": 0.60,
                    "piece_macro_f1": 0.50,
                    "legality_average_precision": 0.45,
                },
            },
        ]
    )

    assert [payload["epoch"] for payload in pareto_epochs] == [1, 2]
