from pathlib import Path

from modalchess.eval.eval_baseline import run_evaluation


def test_eval_smoke_run(tmp_path: Path) -> None:
    config = {
        "output_dir": str(tmp_path),
        "dataset": {"history_length": 1},
        "model": {
            "history_length": 1,
            "input_channels": 18,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 2,
            "dropout": 0.0,
            "use_relation_bias": True,
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
        "metrics": {"topk": [1, 3, 5]},
    }
    metrics = run_evaluation(config)
    assert "piece_occupancy_accuracy" in metrics
    assert "top_1_move_accuracy" in metrics
    assert Path(metrics["report_json"]).exists()
    assert Path(metrics["report_csv"]).exists()
