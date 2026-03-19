"""지도학습 spatial baseline 학습 엔트리포인트."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from modalchess.data.collators import collate_position_samples
from modalchess.data.dataset_builder import DatasetBuildConfig, build_fixture_dataset
from modalchess.models.modalchess_core import ModalChessCoreModel
from modalchess.train.optim import build_optimizer
from modalchess.train.trainer import Trainer
from modalchess.utils.config import load_yaml_config
from modalchess.utils.device import resolve_device
from modalchess.utils.seed import seed_everything


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
        concept_vocab=model_config.get("concept_vocab", []),
    )


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """베이스라인 학습을 실행하고 요약 지표를 반환한다."""
    seed_everything(int(config.get("seed", 7)))
    dataset_config = DatasetBuildConfig(**config.get("dataset", {}))
    dataset = build_fixture_dataset(dataset_config)
    concept_vocab = config["model"].get("concept_vocab", [])
    dataloader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=lambda samples: collate_position_samples(samples, concept_vocab=concept_vocab),
    )
    model = build_model_from_config(config["model"])
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

    save(model.state_dict(), checkpoint_path)
    metrics = {
        "epoch_metrics": epoch_metrics,
        "overfit_metrics": overfit_metrics,
        "checkpoint_path": str(checkpoint_path),
        "num_samples": len(dataset),
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
