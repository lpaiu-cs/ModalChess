"""Week-4 supervised reference-artifact locking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modalchess.utils.config import load_yaml_config, write_yaml_config


REFERENCE_VARIANTS = {
    "g1": {
        "label": "policy_plus_state",
        "experiment_dir": "exp3_ground_state",
    },
    "g3": {
        "label": "policy_plus_state_plus_legality",
        "experiment_dir": "exp3_ground_state_legality",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_summary_markdown(reference_rows: list[dict[str, Any]]) -> str:
    lines = ["# Week-4 Reference Artifacts", ""]
    for row in reference_rows:
        lines.append(
            f"- `{row['variant']}` seed `{row['seed']}`: "
            f"val_nll={row['val_target_move_nll']:.6f}, "
            f"test_nll={row['test_target_move_nll']:.6f}, "
            f"test_top1={row['test_top_1']:.6f}, "
            f"checkpoint=`{row['checkpoint_path']}`"
        )
    lines.append("")
    lines.append("G1 remains the primary supervised backbone. G3 remains the legality-aware control.")
    return "\n".join(lines) + "\n"


def lock_week4_reference_artifacts(
    *,
    week3_root: str | Path = "outputs/week3",
    output_dir: str | Path = "outputs/week4/reference_artifacts",
    train_dataset_path: str | Path = "data/pilot/real_v1/supervised_train.jsonl",
    val_dataset_path: str | Path = "data/pilot/real_v1/supervised_val.jsonl",
    test_dataset_path: str | Path = "data/pilot/real_v1/supervised_test.jsonl",
    puzzle_eval_path: str | Path = "data/pilot/real_v1/puzzle_eval.jsonl",
    seeds: tuple[int, ...] = (11, 17, 23),
) -> dict[str, Any]:
    """Write week-4 reference metadata for frozen G1/G3 artifacts."""
    root = Path(week3_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    reference_rows: list[dict[str, Any]] = []
    for variant_name, variant_info in REFERENCE_VARIANTS.items():
        for seed in seeds:
            run_dir = root / variant_info["experiment_dir"] / f"seed{seed}"
            run_metadata = _load_json(run_dir / "run_metadata.json")
            eval_summary = _load_json(run_dir / "eval_main" / "eval_summary.json")
            metadata = {
                "variant": variant_name,
                "variant_label": variant_info["label"],
                "seed": seed,
                "commit_hash": run_metadata["git_hash"],
                "config_path": str(run_dir / "train_config.yaml"),
                "eval_config_path": str(run_dir / "eval_main" / "eval_config.yaml"),
                "model_parameter_count": run_metadata["model_parameter_count"],
                "checkpoint_path": run_metadata["best_checkpoint_path"],
                "train_dataset_path": run_metadata.get("train_dataset_path", str(train_dataset_path)),
                "val_dataset_path": run_metadata.get("val_dataset_path", str(val_dataset_path)),
                "test_dataset_path": str(test_dataset_path),
                "puzzle_eval_path": run_metadata.get("puzzle_eval_path", str(puzzle_eval_path)),
                "val_target_move_nll": eval_summary["val_target_move_nll"],
                "val_top_1": eval_summary["val_top_1"],
                "test_target_move_nll": eval_summary["test_target_move_nll"],
                "test_top_1": eval_summary["test_top_1"],
                "test_top_3": eval_summary["test_top_3"],
                "test_top_5": eval_summary["test_top_5"],
                "test_occupied_square_accuracy": eval_summary["test_occupied_square_accuracy"],
                "test_piece_macro_f1": eval_summary["test_piece_macro_f1"],
                "test_legality_average_precision": eval_summary["test_legality_average_precision"],
                "test_legality_f1": eval_summary["test_legality_f1"],
            }
            metadata_path = destination / f"{variant_name}_seed{seed}_metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            write_yaml_config(destination / f"{variant_name}_seed{seed}_train_config.yaml", load_yaml_config(run_dir / "train_config.yaml"))
            write_yaml_config(
                destination / f"{variant_name}_seed{seed}_eval_config.yaml",
                load_yaml_config(run_dir / "eval_main" / "eval_config.yaml"),
            )
            reference_rows.append(metadata)

    summary_path = destination / "reference_summary.md"
    summary_path.write_text(_variant_summary_markdown(reference_rows), encoding="utf-8")
    index_path = destination / "reference_index.json"
    index_path.write_text(json.dumps({"artifacts": reference_rows}, indent=2), encoding="utf-8")
    return {
        "reference_rows": reference_rows,
        "summary_path": str(summary_path),
        "index_path": str(index_path),
    }
