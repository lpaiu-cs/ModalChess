"""베이스라인 평가 엔트리포인트."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from modalchess.eval.pipeline import (
    build_eval_dataloader,
    evaluate_model_on_dataloader,
    resolve_named_eval_dataset_configs,
)
from modalchess.eval.report import write_failure_dump, write_report
from modalchess.train.train_spatial_baseline import build_model_from_config, resolve_model_config
from modalchess.utils.config import deep_merge_dict, load_yaml_config, write_yaml_config
from modalchess.utils.device import resolve_device


def _load_checkpoint(checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device)


def _resolve_model_config_from_eval(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    if config.get("model") or config.get("model_config_paths"):
        return resolve_model_config(config)
    if "resolved_model_config" not in checkpoint:
        raise ValueError("checkpoint에 resolved_model_config가 없어 eval model 구성이 불가능하다.")
    return checkpoint["resolved_model_config"]


def _build_summary_payload(
    split_results: dict[str, dict[str, object]],
    checkpoint_path: str,
    checkpoint: dict[str, Any],
    model_type: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
        "seed": checkpoint.get("seed"),
        "git_hash": checkpoint.get("git_hash", "unknown"),
        "model_type": model_type,
        "splits": split_results,
    }
    for split_name, metrics in split_results.items():
        for metric_name in (
            "target_move_nll",
            "top_1",
            "top_3",
            "top_5",
            "occupied_square_accuracy",
            "piece_macro_f1",
            "legality_average_precision",
            "legality_f1",
        ):
            if metric_name in metrics:
                summary[f"{split_name}_{metric_name}"] = metrics[metric_name]
    return summary


def run_evaluation(
    config: dict[str, Any],
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """checkpoint를 불러와 split별 평가를 실행한다."""
    if checkpoint_path is None:
        raise ValueError("평가는 학습된 checkpoint를 명시적으로 받아야 한다.")
    device = resolve_device()
    output_dir = Path(config.get("output_dir", "outputs/eval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(config.get("metrics", {}).get("batch_size", 32))
    topk = list(config.get("metrics", {}).get("topk", [1, 3, 5]))
    named_dataset_configs = resolve_named_eval_dataset_configs(config)
    if not named_dataset_configs:
        raise ValueError("평가 대상 split이 비어 있다.")
    checkpoint: dict[str, Any] | None = None
    if config.get("model") or config.get("model_config_paths"):
        model_config = resolve_model_config(config)
    else:
        checkpoint = _load_checkpoint(checkpoint_path, device=device)
        model_config = _resolve_model_config_from_eval(config, checkpoint)

    concept_vocab = model_config.get("concept_vocab", [])
    prepared_dataloaders: dict[str, tuple[object, object]] = {}
    for split_name, dataset_config in named_dataset_configs.items():
        prepared_dataloaders[split_name] = build_eval_dataloader(
            dataset_config,
            batch_size=batch_size,
            concept_vocab=concept_vocab,
            fen_max_length=model_config.get("max_fen_length"),
            shuffle=False,
        )

    if checkpoint is None:
        checkpoint = _load_checkpoint(checkpoint_path, device=device)

    model = build_model_from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    write_yaml_config(output_dir / "eval_config.yaml", config)
    write_yaml_config(output_dir / "resolved_model_config.yaml", model_config)

    split_results: dict[str, dict[str, object]] = {}

    for split_name, (dataset, dataloader) in prepared_dataloaders.items():
        metrics, prediction_rows = evaluate_model_on_dataloader(
            model,
            dataloader,
            topk=topk,
            device=device,
        )
        failure_rows = []
        for row in prediction_rows:
            enriched = dict(row)
            enriched["split"] = split_name
            enriched["seed"] = checkpoint.get("seed")
            enriched["model_type"] = model_config.get("architecture", "spatial")
            if not row["is_correct_top_1"]:
                failure_rows.append(enriched)
        split_output_dir = output_dir / split_name
        report_paths = write_report(metrics, output_dir=split_output_dir, name="eval_report")
        failure_dump_path = write_failure_dump(
            failure_rows,
            output_dir=split_output_dir,
            name="eval_failures",
        )
        metrics["report_json"] = report_paths["json"]
        metrics["report_csv"] = report_paths["csv"]
        metrics["failure_dump_jsonl"] = failure_dump_path
        metrics["num_failures"] = len(failure_rows)
        metrics["num_samples"] = len(dataset)
        split_results[split_name] = metrics

    model_type = str(model_config.get("architecture", "spatial"))
    summary = _build_summary_payload(
        split_results,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        model_type=model_type,
    )
    summary_paths = write_report(summary, output_dir=output_dir, name="eval_summary")
    summary["summary_json"] = summary_paths["json"]
    summary["summary_csv"] = summary_paths["csv"]
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_path": checkpoint_path,
                "seed": checkpoint.get("seed"),
                "git_hash": checkpoint.get("git_hash", "unknown"),
                "model_type": model_type,
                "eval_config_path": str(output_dir / "eval_config.yaml"),
                "resolved_model_config_path": str(output_dir / "resolved_model_config.yaml"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if len(split_results) == 1:
        single_split_name, single_split_metrics = next(iter(split_results.items()))
        single_result = dict(summary)
        single_result["split"] = single_split_name
        single_result.update(single_split_metrics)
        return single_result
    return summary


def parse_args() -> argparse.Namespace:
    """평가용 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--val-dataset-path", default=None)
    parser.add_argument("--test-dataset-path", default=None)
    return parser.parse_args()


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    overridden = deep_merge_dict(config, {})
    if args.output_dir is not None:
        overridden["output_dir"] = args.output_dir
    if args.dataset_path is not None:
        dataset_section = dict(overridden.get("dataset", {}))
        dataset_section["dataset_path"] = args.dataset_path
        overridden["dataset"] = dataset_section
    if args.val_dataset_path is not None:
        datasets = dict(overridden.get("datasets", {}))
        val_section = dict(datasets.get("val", overridden.get("dataset", {})))
        val_section["dataset_path"] = args.val_dataset_path
        datasets["val"] = val_section
        overridden["datasets"] = datasets
    if args.test_dataset_path is not None:
        datasets = dict(overridden.get("datasets", {}))
        test_section = dict(datasets.get("test", overridden.get("dataset", {})))
        test_section["dataset_path"] = args.test_dataset_path
        datasets["test"] = test_section
        overridden["datasets"] = datasets
    return overridden


def main() -> None:
    """CLI 엔트리포인트."""
    args = parse_args()
    config = _apply_cli_overrides(load_yaml_config(args.config), args)
    run_evaluation(config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
