"""베이스라인 평가 엔트리포인트."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch.utils.data import DataLoader

from modalchess.data.collators import collate_position_samples
from modalchess.data.dataset_builder import DatasetBuildConfig, build_dataset
from modalchess.eval.metrics_move_quality import compute_move_quality_metrics
from modalchess.eval.metrics_state_fidelity import compute_state_fidelity_metrics
from modalchess.eval.report import write_report
from modalchess.train.train_spatial_baseline import build_model_from_config, resolve_model_config
from modalchess.train.trainer import move_batch_to_device
from modalchess.utils.config import load_yaml_config
from modalchess.utils.device import autocast_context, resolve_device


@torch.no_grad()
def run_evaluation(
    config: dict[str, Any],
    checkpoint_path: str | None = None,
) -> dict[str, float]:
    """로컬 fixture 데이터로 평가를 실행한다."""
    if checkpoint_path is None:
        raise ValueError("평가는 학습된 checkpoint를 명시적으로 받아야 한다.")
    dataset_config = DatasetBuildConfig(**config.get("dataset", {}))
    dataset = build_dataset(dataset_config)
    model_config = resolve_model_config(config)
    concept_vocab = model_config.get("concept_vocab", [])
    dataloader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        collate_fn=lambda samples: collate_position_samples(samples, concept_vocab=concept_vocab),
    )
    device = resolve_device()
    model = build_model_from_config(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    batch = next(iter(dataloader))
    device_batch = move_batch_to_device(batch, device)
    with autocast_context(device):
        outputs = model(device_batch["board_planes"], meta_features=device_batch["meta_features"])
    metrics = {}
    metrics.update(compute_state_fidelity_metrics(outputs, device_batch))
    metrics.update(compute_move_quality_metrics(outputs, batch, topk=config.get("metrics", {}).get("topk", [1, 3, 5])))
    output_dir = config.get("output_dir", "outputs/eval")
    report_paths = write_report(metrics, output_dir=output_dir)
    metrics["report_json"] = report_paths["json"]
    metrics["report_csv"] = report_paths["csv"]
    return metrics


def parse_args() -> argparse.Namespace:
    """평가용 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    """CLI 엔트리포인트."""
    args = parse_args()
    config = load_yaml_config(args.config)
    run_evaluation(config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
