"""공통 train/eval 경로에서 재사용하는 평가 파이프라인."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from modalchess.data.collators import collate_position_samples
from modalchess.data.dataset_builder import DatasetBuildConfig, build_dataset
from modalchess.eval.metrics_move_quality import collect_move_prediction_rows, summarize_move_prediction_rows
from modalchess.eval.metrics_state_fidelity import StateFidelityAccumulator
from modalchess.train.trainer import move_batch_to_device
from modalchess.utils.config import deep_merge_dict
from modalchess.utils.device import autocast_context, resolve_device


def build_eval_dataloader(
    dataset_config: DatasetBuildConfig,
    batch_size: int,
    concept_vocab: list[str],
    fen_max_length: int | None = None,
    shuffle: bool = False,
) -> tuple[object, DataLoader]:
    """평가용 데이터셋과 DataLoader를 함께 생성한다."""
    dataset = build_dataset(dataset_config)
    if len(dataset) == 0:
        raise ValueError("평가 데이터셋이 비어 있다. split/ratio/dataset_path 설정을 확인해야 한다.")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda samples: collate_position_samples(
            samples,
            concept_vocab=concept_vocab,
            fen_max_length=fen_max_length,
        ),
    )
    return dataset, dataloader


def evaluate_model_on_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    topk: list[int],
    device: torch.device | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """단일 dataloader에 대해 상태/수 예측 지표를 계산한다."""
    target_device = device or resolve_device()
    state_accumulator = StateFidelityAccumulator()
    prediction_rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            device_batch = move_batch_to_device(batch, target_device)
            with autocast_context(target_device):
                outputs = model(
                    board_planes=device_batch.get("board_planes"),
                    meta_features=device_batch.get("meta_features"),
                    fen_token_ids=device_batch.get("fen_token_ids"),
                    fen_attention_mask=device_batch.get("fen_attention_mask"),
                )
            state_accumulator.update(outputs, device_batch)
            prediction_rows.extend(collect_move_prediction_rows(outputs, batch, topk=topk))
    metrics: dict[str, object] = {}
    metrics.update(state_accumulator.compute())
    metrics.update(summarize_move_prediction_rows(prediction_rows, topk=topk))
    return metrics, prediction_rows


def resolve_named_eval_dataset_configs(config: dict[str, Any]) -> dict[str, DatasetBuildConfig]:
    """평가 config에서 split 이름별 데이터셋 설정을 구성한다."""
    if "datasets" in config:
        return {
            split_name: DatasetBuildConfig(**dataset_config)
            for split_name, dataset_config in config["datasets"].items()
        }
    base_dataset_config = config.get("dataset", {})
    if "evaluation_splits" in config:
        return {
            split_name: DatasetBuildConfig(
                **deep_merge_dict(base_dataset_config, {"split": split_name})
            )
            for split_name in config["evaluation_splits"]
        }
    split_name = base_dataset_config.get("split", "all")
    return {split_name: DatasetBuildConfig(**base_dataset_config)}
