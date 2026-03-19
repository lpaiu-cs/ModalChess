from pathlib import Path

from modalchess.eval.eval_baseline import run_evaluation
from modalchess.train.train_spatial_baseline import run_training


def test_eval_smoke_run(tmp_path: Path) -> None:
    train_output = tmp_path / "train"
    eval_output = tmp_path / "eval"
    train_config = {
        "seed": 7,
        "output_dir": str(train_output),
        "dataset": {"source": "fixture", "history_length": 1, "limit": 2},
        "model": {
            "history_length": 1,
            "input_channels": 18,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 2,
            "dropout": 0.0,
            "use_relation_bias": True,
            "legality_hidden_dim": 32,
            "use_pair_scorer": False,
            "meta_num_tokens": 2,
            "concept_vocab": [
                "check",
                "capture",
                "recapture",
                "pin",
                "fork",
                "skewer",
                "discovered_attack",
                "discovered_check",
                "king_safety",
                "passed_pawn",
                "open_file",
                "promotion_threat",
            ],
        },
        "train": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "overfit_steps": 2,
        },
        "losses": {
            "policy": 1.0,
            "policy_axis_ce": 1.0,
            "policy_listwise": 1.0,
            "state_probe": 1.0,
            "legality": 0.25,
            "value": 0.1,
            "concept": 0.1,
        },
    }
    training_metrics = run_training(train_config)
    eval_config = {
        "output_dir": str(eval_output),
        "dataset": {"source": "fixture", "history_length": 1, "limit": 2},
        "model": train_config["model"],
        "metrics": {"topk": [1, 3, 5]},
    }
    metrics = run_evaluation(eval_config, checkpoint_path=training_metrics["checkpoint_path"])
    assert "square_state_accuracy" in metrics
    assert "top_1_move_accuracy" in metrics
    assert "target_move_nll" in metrics
    assert Path(metrics["report_json"]).exists()
    assert Path(metrics["report_csv"]).exists()
    assert Path(metrics["failure_dump_jsonl"]).exists()
