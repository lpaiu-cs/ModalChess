"""베이스라인 평가 엔트리포인트."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch.utils.data import DataLoader

from modalchess.data.collators import collate_position_samples
from modalchess.data.dataset_builder import DatasetBuildConfig, build_dataset
from modalchess.eval.metrics_move_quality import collect_move_prediction_rows, summarize_move_prediction_rows
from modalchess.eval.metrics_state_fidelity import StateFidelityAccumulator
from modalchess.eval.report import write_failure_dump, write_report
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
    if dataset_config.source != "fixture" and dataset_config.split not in {"val", "test"}:
        raise ValueError("JSONL 평가에서는 split을 val 또는 test로 명시해야 한다.")
    dataset = build_dataset(dataset_config)
    if len(dataset) == 0:
        raise ValueError("평가 데이터셋이 비어 있다. split/ratio/dataset_path 설정을 확인해야 한다.")
    model_config = resolve_model_config(config)
    concept_vocab = model_config.get("concept_vocab", [])
    batch_size = int(config.get("metrics", {}).get("batch_size", min(len(dataset), 64)))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda samples: collate_position_samples(
            samples,
            concept_vocab=concept_vocab,
            fen_max_length=model_config.get("max_fen_length"),
        ),
    )
    device = resolve_device()
    model = build_model_from_config(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    topk = config.get("metrics", {}).get("topk", [1, 3, 5])
    state_accumulator = StateFidelityAccumulator()
    prediction_rows: list[dict[str, object]] = []
    for batch in dataloader:
        device_batch = move_batch_to_device(batch, device)
        with autocast_context(device):
            outputs = model(
                board_planes=device_batch.get("board_planes"),
                meta_features=device_batch.get("meta_features"),
                fen_token_ids=device_batch.get("fen_token_ids"),
                fen_attention_mask=device_batch.get("fen_attention_mask"),
            )
        state_accumulator.update(outputs, device_batch)
        prediction_rows.extend(collect_move_prediction_rows(outputs, batch, topk=topk))
    metrics = {}
    metrics.update(state_accumulator.compute())
    metrics.update(summarize_move_prediction_rows(prediction_rows, topk=topk))
    failure_rows = [row for row in prediction_rows if not row["is_correct_top_1"]]
    output_dir = config.get("output_dir", "outputs/eval")
    report_paths = write_report(metrics, output_dir=output_dir)
    failure_dump_path = write_failure_dump(failure_rows, output_dir=output_dir)
    metrics["report_json"] = report_paths["json"]
    metrics["report_csv"] = report_paths["csv"]
    metrics["failure_dump_jsonl"] = failure_dump_path
    metrics["num_failures"] = float(len(failure_rows))
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
