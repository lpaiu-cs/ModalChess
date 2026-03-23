"""지도학습 spatial/FEN baseline 학습 엔트리포인트."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any

import torch
from torch.utils.data import DataLoader

from modalchess.data.collators import collate_position_samples
from modalchess.data.dataset_builder import DatasetBuildConfig, build_dataset
from modalchess.data.fen_tokenizer import FenTokenizer
from modalchess.eval.pipeline import build_eval_dataloader, evaluate_model_on_dataloader
from modalchess.models.fen_baseline import FenPolicyBaselineModel
from modalchess.models.modalchess_core import ModalChessCoreModel
from modalchess.train.optim import build_optimizer
from modalchess.train.trainer import Trainer
from modalchess.utils.config import (
    deep_merge_dict,
    load_and_merge_yaml_configs,
    load_yaml_config,
    write_yaml_config,
)
from modalchess.utils.device import resolve_device
from modalchess.utils.seed import seed_everything


POLICY_SELECTION_METRIC = "val.target_move_nll"
GROUNDING_SELECTION_COMPONENTS = (
    "occupied_square_accuracy",
    "piece_macro_f1",
    "legality_average_precision",
)
GROUNDING_SELECTION_METRIC = (
    "mean(val.occupied_square_accuracy,val.piece_macro_f1,val.legality_average_precision)"
)


def resolve_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """공통 model/head 설정과 실행별 override를 병합한다."""
    base_model_config = load_and_merge_yaml_configs(config.get("model_config_paths", []))
    return deep_merge_dict(base_model_config, config.get("model", {}))


def resolve_git_hash() -> str:
    """현재 저장소의 git hash를 반환한다."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_model_from_config(model_config: dict[str, Any]) -> ModalChessCoreModel:
    """설정 딕셔너리로부터 코어 모델을 생성한다."""
    architecture = model_config.get("architecture", "spatial")
    if architecture == "fen":
        tokenizer = FenTokenizer.default()
        return FenPolicyBaselineModel(
            vocab_size=len(tokenizer.vocab),
            max_length=model_config.get("max_fen_length", 128),
            d_model=model_config["d_model"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            dropout=model_config.get("dropout", 0.1),
            legality_hidden_dim=model_config.get("legality_hidden_dim", 64),
            concept_vocab=model_config.get("concept_vocab", []),
            use_pair_scorer=model_config.get("use_pair_scorer", False),
            meta_num_tokens=model_config.get("meta_num_tokens", 2),
            meta_hidden_dim=model_config.get("meta_hidden_dim"),
            policy_pool=model_config.get("policy_pool", "context"),
            state_probe_pool=model_config.get("state_probe_pool", "context"),
            value_pool=model_config.get("value_pool", "context"),
            concept_pool=model_config.get("concept_pool", "context"),
        )
    return ModalChessCoreModel(
        history_length=model_config["history_length"],
        input_channels=model_config["input_channels"],
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        mlp_ratio=model_config.get("mlp_ratio", 4),
        dropout=model_config.get("dropout", 0.1),
        use_relation_bias=model_config.get("use_relation_bias", True),
        legality_hidden_dim=model_config.get("legality_hidden_dim", 64),
        concept_vocab=model_config.get("concept_vocab", []),
        use_pair_scorer=model_config.get("use_pair_scorer", False),
        meta_num_tokens=model_config.get("meta_num_tokens", 2),
        meta_hidden_dim=model_config.get("meta_hidden_dim"),
        policy_pool=model_config.get("policy_pool", "context"),
        state_probe_pool=model_config.get("state_probe_pool", "context"),
        value_pool=model_config.get("value_pool", "context"),
        concept_pool=model_config.get("concept_pool", "context"),
        )


def count_model_parameters(model: torch.nn.Module) -> dict[str, int]:
    """모델의 전체/학습가능 파라미터 수를 계산한다."""
    return {
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


def _build_train_dataloader(
    dataset_config: DatasetBuildConfig,
    batch_size: int,
    concept_vocab: list[str],
    fen_max_length: int | None,
) -> tuple[object, DataLoader]:
    dataset = build_dataset(dataset_config)
    if len(dataset) == 0:
        raise ValueError("학습 데이터셋이 비어 있다. split/ratio/dataset_path 설정을 확인해야 한다.")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda samples: collate_position_samples(
            samples,
            concept_vocab=concept_vocab,
            fen_max_length=fen_max_length,
        ),
    )
    return dataset, dataloader


def _resolve_dataset_section(
    config: dict[str, Any],
    section_name: str,
    fallback_split: str | None = None,
) -> dict[str, Any] | None:
    if section_name in config:
        return config[section_name]
    if section_name != "dataset":
        base_dataset = config.get("dataset")
        if base_dataset is None:
            return None
        if fallback_split is None:
            return base_dataset
        if base_dataset.get("source") == "jsonl":
            return deep_merge_dict(base_dataset, {"split": fallback_split})
        if fallback_split == "train":
            return base_dataset
        return None
    return config.get("dataset")


def _build_checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    model_config: dict[str, Any],
    seed: int,
    git_hash: str,
    epoch_metrics: list[dict[str, Any]],
    selection_metric: str,
    best_epoch: int | None,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "resolved_model_config": model_config,
        "seed": seed,
        "git_hash": git_hash,
        "epoch_metrics": epoch_metrics,
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
    }


def _extract_grounding_components(val_metrics: dict[str, Any]) -> dict[str, float]:
    missing = [metric_name for metric_name in GROUNDING_SELECTION_COMPONENTS if metric_name not in val_metrics]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "grounding-best selection에는 val metrics가 모두 필요하다. "
            f"누락된 항목: {missing_text}"
        )
    return {
        metric_name: float(val_metrics[metric_name])
        for metric_name in GROUNDING_SELECTION_COMPONENTS
    }


def _compute_grounding_score(val_metrics: dict[str, Any]) -> float:
    components = _extract_grounding_components(val_metrics)
    return sum(components.values()) / len(components)


def _pareto_epoch_payload(epoch_record: dict[str, Any]) -> dict[str, Any]:
    val_metrics = epoch_record.get("val")
    if val_metrics is None:
        raise ValueError("Pareto summary에는 val metrics가 필요하다.")
    components = _extract_grounding_components(val_metrics)
    return {
        "epoch": int(epoch_record["epoch"]),
        "target_move_nll": float(val_metrics["target_move_nll"]),
        **components,
        "grounding_score": sum(components.values()) / len(components),
    }


def _is_pareto_dominated(candidate: dict[str, Any], challenger: dict[str, Any]) -> bool:
    not_worse = (
        challenger["target_move_nll"] <= candidate["target_move_nll"]
        and challenger["occupied_square_accuracy"] >= candidate["occupied_square_accuracy"]
        and challenger["piece_macro_f1"] >= candidate["piece_macro_f1"]
        and challenger["legality_average_precision"] >= candidate["legality_average_precision"]
    )
    strictly_better = (
        challenger["target_move_nll"] < candidate["target_move_nll"]
        or challenger["occupied_square_accuracy"] > candidate["occupied_square_accuracy"]
        or challenger["piece_macro_f1"] > candidate["piece_macro_f1"]
        or challenger["legality_average_precision"] > candidate["legality_average_precision"]
    )
    return not_worse and strictly_better


def _compute_pareto_epochs(epoch_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluated_epochs = [_pareto_epoch_payload(epoch_record) for epoch_record in epoch_metrics if "val" in epoch_record]
    pareto_epochs: list[dict[str, Any]] = []
    for candidate in evaluated_epochs:
        if any(_is_pareto_dominated(candidate, challenger) for challenger in evaluated_epochs):
            continue
        pareto_epochs.append(candidate)
    return sorted(pareto_epochs, key=lambda payload: payload["epoch"])


def _write_selection_summary_csv(selection_rows: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "epoch",
        "has_val_metrics",
        "target_move_nll",
        "occupied_square_accuracy",
        "piece_macro_f1",
        "legality_average_precision",
        "grounding_score",
        "is_policy_best",
        "is_grounding_best",
        "is_pareto",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selection_rows:
            writer.writerow(row)


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """베이스라인 학습을 실행하고 요약 지표를 반환한다."""
    seed = int(config.get("seed", 7))
    seed_everything(seed)

    model_config = resolve_model_config(config)
    concept_vocab = model_config.get("concept_vocab", [])
    fen_max_length = model_config.get("max_fen_length")
    train_dataset_section = _resolve_dataset_section(
        config,
        "train_dataset",
        fallback_split="train",
    )
    if train_dataset_section is None:
        raise ValueError("학습에는 dataset 또는 train_dataset 설정이 필요하다.")
    train_dataset_config = DatasetBuildConfig(**train_dataset_section)
    val_dataset_section = _resolve_dataset_section(config, "val_dataset", fallback_split="val")

    batch_size = int(config["train"]["batch_size"])
    train_dataset, train_dataloader = _build_train_dataloader(
        train_dataset_config,
        batch_size=batch_size,
        concept_vocab=concept_vocab,
        fen_max_length=fen_max_length,
    )
    val_dataset = None
    val_dataloader = None
    if val_dataset_section is not None:
        val_dataset_config = DatasetBuildConfig(**val_dataset_section)
        val_batch_size = int(config["train"].get("eval_batch_size", batch_size))
        val_dataset, val_dataloader = build_eval_dataloader(
            val_dataset_config,
            batch_size=val_batch_size,
            concept_vocab=concept_vocab,
            fen_max_length=fen_max_length,
            shuffle=False,
        )

    model = build_model_from_config(model_config)
    parameter_counts = count_model_parameters(model)
    optimizer = build_optimizer(
        model=model,
        learning_rate=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    device = resolve_device()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_weights=config.get("losses", {}),
        device=device,
        grad_clip_norm=config["train"].get("grad_clip_norm"),
    )

    output_dir = Path(config.get("output_dir", "outputs/train"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml_config(output_dir / "train_config.yaml", config)
    write_yaml_config(output_dir / "resolved_model_config.yaml", model_config)

    git_hash = resolve_git_hash()
    policy_selection_metric = POLICY_SELECTION_METRIC
    grounding_selection_metric = GROUNDING_SELECTION_METRIC
    best_policy_value = float("inf")
    best_grounding_value = float("-inf")
    best_epoch: int | None = None
    grounding_best_epoch: int | None = None
    best_metrics: dict[str, object] | None = None
    best_checkpoint_path = output_dir / "best_model.pt"
    best_policy_checkpoint_path = output_dir / "best_policy_model.pt"
    best_grounding_checkpoint_path = output_dir / "best_grounding_model.pt"
    last_checkpoint_path = output_dir / "last_model.pt"
    selection_summary_json_path = output_dir / "selection_summary.json"
    selection_summary_csv_path = output_dir / "selection_summary.csv"
    pareto_epochs_path = output_dir / "pareto_epochs.json"
    epoch_metrics: list[dict[str, Any]] = []
    epochs = int(config["train"]["epochs"])
    validation_topk = list(config.get("metrics", {}).get("topk", [1, 3, 5]))
    grounding_selection_enabled = val_dataloader is not None
    best_grounding_metrics: dict[str, object] | None = None

    for epoch_index in range(epochs):
        train_metrics = trainer.train_epoch(train_dataloader)
        epoch_record: dict[str, Any] = {
            "epoch": epoch_index + 1,
            "train": train_metrics,
        }
        if val_dataloader is not None:
            val_metrics, _ = evaluate_model_on_dataloader(
                trainer.model,
                val_dataloader,
                topk=validation_topk,
                device=device,
            )
            epoch_record["val"] = val_metrics
            current_policy_value = float(val_metrics["target_move_nll"])
            if current_policy_value < best_policy_value:
                best_policy_value = current_policy_value
                best_epoch = epoch_index + 1
                best_metrics = val_metrics
                policy_payload = _build_checkpoint_payload(
                    trainer.model,
                    optimizer,
                    config,
                    model_config,
                    seed,
                    git_hash,
                    epoch_metrics + [epoch_record],
                    selection_metric=policy_selection_metric,
                    best_epoch=best_epoch,
                )
                torch.save(policy_payload, best_checkpoint_path)
                torch.save(policy_payload, best_policy_checkpoint_path)
            if grounding_selection_enabled:
                current_grounding_value = _compute_grounding_score(val_metrics)
                epoch_record["grounding_score"] = current_grounding_value
                if current_grounding_value > best_grounding_value:
                    best_grounding_value = current_grounding_value
                    grounding_best_epoch = epoch_index + 1
                    best_grounding_metrics = val_metrics
                    torch.save(
                        _build_checkpoint_payload(
                            trainer.model,
                            optimizer,
                            config,
                            model_config,
                            seed,
                            git_hash,
                            epoch_metrics + [epoch_record],
                            selection_metric=grounding_selection_metric,
                            best_epoch=grounding_best_epoch,
                        ),
                        best_grounding_checkpoint_path,
                    )
        epoch_metrics.append(epoch_record)

    torch.save(
        _build_checkpoint_payload(
            trainer.model,
            optimizer,
            config,
            model_config,
            seed,
            git_hash,
            epoch_metrics,
            selection_metric=policy_selection_metric,
            best_epoch=best_epoch,
        ),
        last_checkpoint_path,
    )
    if not best_checkpoint_path.exists():
        torch.save(
            _build_checkpoint_payload(
                trainer.model,
                optimizer,
                config,
                model_config,
                seed,
                git_hash,
                epoch_metrics,
                selection_metric=policy_selection_metric,
                best_epoch=epochs,
            ),
            best_checkpoint_path,
        )
        torch.save(
            _build_checkpoint_payload(
                trainer.model,
                optimizer,
                config,
                model_config,
                seed,
                git_hash,
                epoch_metrics,
                selection_metric=policy_selection_metric,
                best_epoch=epochs,
            ),
            best_policy_checkpoint_path,
        )
        best_epoch = epochs

    if grounding_selection_enabled and not best_grounding_checkpoint_path.exists():
        raise ValueError("grounding-best selection이 활성화됐지만 checkpoint를 만들지 못했다.")

    pareto_epochs = _compute_pareto_epochs(epoch_metrics) if grounding_selection_enabled else []
    pareto_epochs_path.write_text(json.dumps(pareto_epochs, indent=2), encoding="utf-8")
    selection_rows: list[dict[str, Any]] = []
    pareto_epoch_numbers = {payload["epoch"] for payload in pareto_epochs}
    for epoch_record in epoch_metrics:
        val_metrics = epoch_record.get("val")
        row: dict[str, Any] = {
            "epoch": int(epoch_record["epoch"]),
            "has_val_metrics": val_metrics is not None,
            "target_move_nll": "",
            "occupied_square_accuracy": "",
            "piece_macro_f1": "",
            "legality_average_precision": "",
            "grounding_score": "",
            "is_policy_best": epoch_record["epoch"] == best_epoch,
            "is_grounding_best": epoch_record["epoch"] == grounding_best_epoch,
            "is_pareto": epoch_record["epoch"] in pareto_epoch_numbers,
        }
        if val_metrics is not None:
            row.update(
                {
                    "target_move_nll": float(val_metrics["target_move_nll"]),
                    "occupied_square_accuracy": float(val_metrics["occupied_square_accuracy"]),
                    "piece_macro_f1": float(val_metrics["piece_macro_f1"]),
                    "legality_average_precision": float(val_metrics["legality_average_precision"]),
                    "grounding_score": float(
                        epoch_record.get("grounding_score", _compute_grounding_score(val_metrics))
                    ),
                }
            )
        selection_rows.append(row)

    selection_summary = {
        "policy_selection_metric": policy_selection_metric,
        "grounding_selection_metric": grounding_selection_metric,
        "policy_best_epoch": best_epoch,
        "grounding_best_epoch": grounding_best_epoch,
        "policy_best_value": None if best_epoch is None or best_metrics is None else float(best_metrics["target_move_nll"]),
        "grounding_score": None if grounding_best_epoch is None else float(best_grounding_value),
        "pareto_epochs": pareto_epochs,
        "epochs": selection_rows,
        "policy_best_checkpoint_path": str(best_policy_checkpoint_path),
        "grounding_best_checkpoint_path": (
            str(best_grounding_checkpoint_path) if best_grounding_checkpoint_path.exists() else None
        ),
    }
    selection_summary_json_path.write_text(json.dumps(selection_summary, indent=2), encoding="utf-8")
    _write_selection_summary_csv(selection_rows, selection_summary_csv_path)

    overfit_metrics = None
    if config["train"].get("run_overfit", False) and int(config["train"].get("overfit_steps", 0)) > 0:
        overfit_metrics = trainer.overfit(train_dataloader, int(config["train"]["overfit_steps"]))

    model_type = str(model_config.get("architecture", "spatial"))
    metrics = {
        "epoch_metrics": epoch_metrics,
        "overfit_metrics": overfit_metrics,
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "checkpoint_path": str(best_checkpoint_path),
        "best_policy_checkpoint_path": str(best_policy_checkpoint_path),
        "best_grounding_checkpoint_path": (
            str(best_grounding_checkpoint_path) if best_grounding_checkpoint_path.exists() else None
        ),
        "best_epoch": best_epoch,
        "policy_best_epoch": best_epoch,
        "grounding_best_epoch": grounding_best_epoch,
        "best_val_metrics": best_metrics,
        "best_grounding_val_metrics": best_grounding_metrics,
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset) if val_dataset is not None else 0,
        "seed": seed,
        "git_hash": git_hash,
        "model_type": model_type,
        "selection_metric": policy_selection_metric,
        "policy_selection_metric": policy_selection_metric,
        "grounding_selection_metric": grounding_selection_metric,
        "grounding_score": None if grounding_best_epoch is None else float(best_grounding_value),
        "pareto_epochs": pareto_epochs,
        "selection_summary_json": str(selection_summary_json_path),
        "selection_summary_csv": str(selection_summary_csv_path),
        "pareto_epochs_json": str(pareto_epochs_path),
        **parameter_counts,
    }
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    run_metadata = {
        "seed": seed,
        "git_hash": git_hash,
        "model_type": model_type,
        "selection_metric": policy_selection_metric,
        "policy_best_epoch": best_epoch,
        "grounding_best_epoch": grounding_best_epoch,
        "policy_selection_metric": policy_selection_metric,
        "grounding_selection_metric": grounding_selection_metric,
        "grounding_score": None if grounding_best_epoch is None else float(best_grounding_value),
        "pareto_epochs": [payload["epoch"] for payload in pareto_epochs],
        **parameter_counts,
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_policy_checkpoint_path": str(best_policy_checkpoint_path),
        "best_grounding_checkpoint_path": (
            str(best_grounding_checkpoint_path) if best_grounding_checkpoint_path.exists() else None
        ),
        "last_checkpoint_path": str(last_checkpoint_path),
        "train_config_path": str(output_dir / "train_config.yaml"),
        "resolved_model_config_path": str(output_dir / "resolved_model_config.yaml"),
        "selection_summary_json": str(selection_summary_json_path),
        "selection_summary_csv": str(selection_summary_csv_path),
        "pareto_epochs_json": str(pareto_epochs_path),
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset) if val_dataset is not None else 0,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    """학습용 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/smoke_default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--train-dataset-path", default=None)
    parser.add_argument("--val-dataset-path", default=None)
    return parser.parse_args()


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    overridden = deep_merge_dict(config, {})
    if args.seed is not None:
        overridden["seed"] = args.seed
    if args.output_dir is not None:
        overridden["output_dir"] = args.output_dir
    if args.dataset_path is not None:
        dataset_section = dict(overridden.get("dataset", {}))
        dataset_section["dataset_path"] = args.dataset_path
        overridden["dataset"] = dataset_section
    if args.train_dataset_path is not None:
        train_dataset_section = dict(overridden.get("train_dataset", overridden.get("dataset", {})))
        train_dataset_section["dataset_path"] = args.train_dataset_path
        overridden["train_dataset"] = train_dataset_section
    if args.val_dataset_path is not None:
        val_dataset_section = dict(overridden.get("val_dataset", overridden.get("dataset", {})))
        val_dataset_section["dataset_path"] = args.val_dataset_path
        overridden["val_dataset"] = val_dataset_section
    return overridden


def main() -> None:
    """CLI 엔트리포인트."""
    args = parse_args()
    config = _apply_cli_overrides(load_yaml_config(args.config), args)
    run_training(config)


if __name__ == "__main__":
    main()
