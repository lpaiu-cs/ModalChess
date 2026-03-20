import json
from pathlib import Path

from modalchess.eval.aggregate_week3 import aggregate_week3
from modalchess.eval.metrics_move_quality import summarize_move_prediction_rows


def test_summarize_move_prediction_rows_adds_theme_groups() -> None:
    rows = [
        {
            "target_move": {"src_square": 0, "dst_square": 1, "promotion": 0},
            "top_predictions": [
                {"src_square": 0, "dst_square": 1, "promotion": 0},
            ],
            "target_move_nll": 0.5,
            "is_promotion": False,
            "is_castling": False,
            "is_en_passant": False,
            "is_check_evasion": True,
            "is_correct_top_1": True,
            "is_correct_top_3": True,
            "is_correct_top_5": True,
            "theme_promotion_related": True,
            "theme_check_related": True,
            "theme_castling_related": False,
            "theme_en_passant_related": False,
        },
        {
            "target_move": {"src_square": 1, "dst_square": 2, "promotion": 0},
            "top_predictions": [
                {"src_square": 3, "dst_square": 4, "promotion": 0},
            ],
            "target_move_nll": 1.5,
            "is_promotion": False,
            "is_castling": False,
            "is_en_passant": False,
            "is_check_evasion": False,
            "is_correct_top_1": False,
            "is_correct_top_3": False,
            "is_correct_top_5": False,
            "theme_promotion_related": False,
            "theme_check_related": False,
            "theme_castling_related": True,
            "theme_en_passant_related": True,
        },
    ]

    metrics = summarize_move_prediction_rows(rows, topk=(1, 3, 5))

    assert metrics["theme_promotion_related_count"] == 1
    assert metrics["theme_promotion_related_top_1"] == 1.0
    assert metrics["theme_check_related_count"] == 1
    assert metrics["theme_check_related_target_move_nll"] == 0.5
    assert metrics["theme_castling_related_count"] == 1
    assert metrics["theme_castling_related_top_1"] == 0.0
    assert metrics["theme_en_passant_related_count"] == 1


def test_aggregate_week3_merges_main_and_puzzle_outputs(tmp_path: Path) -> None:
    week3_root = tmp_path / "outputs" / "week3"
    main_dir = week3_root / "exp3_ground_policy_only" / "seed11" / "eval_main"
    puzzle_dir = week3_root / "puzzle_aux_eval" / "exp3_ground_policy_only" / "seed11"
    main_dir.mkdir(parents=True)
    puzzle_dir.mkdir(parents=True)

    main_summary = {
        "seed": 11,
        "model_type": "spatial",
        "model_parameter_count": 123,
        "trainable_parameter_count": 123,
        "val_target_move_nll": 2.0,
        "val_top_1": 0.2,
        "val_top_3": 0.4,
        "val_top_5": 0.5,
        "test_target_move_nll": 2.1,
        "test_top_1": 0.25,
        "test_top_3": 0.45,
        "test_top_5": 0.55,
        "test_occupied_square_accuracy": 0.9,
        "test_piece_macro_f1": 0.8,
        "test_legality_average_precision": 0.3,
        "test_legality_f1": 0.2,
        "splits": {
            "test": {
                "subsets": {
                    "check_evasion": {
                        "count": 2,
                        "top_1": 0.5,
                        "top_3": 1.0,
                        "top_5": 1.0,
                        "target_move_nll": 1.2,
                    }
                }
            }
        },
    }
    puzzle_summary = {
        "seed": 11,
        "model_type": "spatial",
        "splits": {
            "puzzle": {
                "target_move_nll": 1.7,
                "top_1": 0.3,
                "top_3": 0.6,
                "top_5": 0.8,
                "subsets": {
                    "theme_check_related": {
                        "count": 5,
                        "top_1": 0.4,
                        "top_3": 0.8,
                        "top_5": 1.0,
                        "target_move_nll": 1.1,
                    }
                },
            }
        },
    }
    (main_dir / "eval_summary.json").write_text(json.dumps(main_summary), encoding="utf-8")
    (puzzle_dir / "eval_summary.json").write_text(json.dumps(puzzle_summary), encoding="utf-8")

    paths = aggregate_week3(week3_root, puzzle_root=week3_root / "puzzle_aux_eval", output_dir=week3_root)

    aggregate_payload = json.loads(Path(paths["aggregate_json"]).read_text(encoding="utf-8"))
    subset_payload = json.loads(Path(paths["subset_json"]).read_text(encoding="utf-8"))

    assert aggregate_payload["runs"][0]["puzzle_nll"] == 1.7
    assert aggregate_payload["runs"][0]["model_parameter_count"] == 123
    subset_names = {row["subset"] for row in subset_payload["runs"]}
    assert "check_evasion" in subset_names
    assert "theme_check_related" in subset_names
