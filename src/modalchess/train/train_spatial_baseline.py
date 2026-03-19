"""지도학습 spatial baseline 학습 엔트리포인트."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from torch.utils.data import DataLoader

from modalchess.data.collators import collate_position_samples
from modalchess.data.dataset_builder import DatasetBuildConfig, build_dataset
from modalchess.models.modalchess_core import ModalChessCoreModel
from modalchess.train.optim import build_optimizer
from modalchess.train.trainer import Trainer
from modalchess.utils.config import deep_merge_dict, load_and_merge_yaml_configs, load_yaml_config
from modalchess.utils.device import resolve_device
from modalchess.utils.seed import seed_everything


def resolve_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """공통 model/head 설정과 실행별 override를 병합한다."""
    base_model_config = load_and_merge_yaml_configs(config.get("model_config_paths", []))
    return deep_merge_dict(base_model_config, config.get("model", {}))


def resolve_git_hash() -> str:
    """현재 저장소의 git hash를 반환한다."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def build_model_from_config(model_config: dict[str, Any]) -> ModalChessCoreModel:
    """설정 딕셔너리로부터 코어 모델을 생성한다."""
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
    )


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """베이스라인 학습을 실행하고 요약 지표를 반환한다."""
    seed = int(config.get("seed", 7))
    seed_everything(seed)
    dataset_config = DatasetBuildConfig(**config.get("dataset", {}))
    dataset = build_dataset(dataset_config)
    model_config = resolve_model_config(config)
    concept_vocab = model_config.get("concept_vocab", [])
    dataloader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=lambda samples: collate_position_samples(samples, concept_vocab=concept_vocab),
    )
    model = build_model_from_config(model_config)
    optimizer = build_optimizer(
        model=model,
        learning_rate=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_weights=config.get("losses", {}),
        device=resolve_device(),
        grad_clip_norm=config["train"].get("grad_clip_norm"),
    )
    epochs = int(config["train"]["epochs"])
    epoch_metrics = []
    for _ in range(epochs):
        epoch_metrics.append(trainer.train_epoch(dataloader))
    overfit_metrics = trainer.overfit(dataloader, int(config["train"].get("overfit_steps", 16)))
    output_dir = Path(config.get("output_dir", "outputs/train"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    metrics_path = output_dir / "train_metrics.json"
    from torch import save

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "resolved_model_config": model_config,
        "seed": seed,
        "git_hash": resolve_git_hash(),
        "epoch_metrics": epoch_metrics,
        "overfit_metrics": overfit_metrics,
    }
    save(checkpoint, checkpoint_path)
    metrics = {
        "epoch_metrics": epoch_metrics,
        "overfit_metrics": overfit_metrics,
        "checkpoint_path": str(checkpoint_path),
        "num_samples": len(dataset),
        "seed": seed,
        "git_hash": checkpoint["git_hash"],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    """학습용 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/default.yaml")
    return parser.parse_args()


def main() -> None:
    """CLI 엔트리포인트."""
    args = parse_args()
    config = load_yaml_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
