"""Frozen-backbone embedding export helpers for week-4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from modalchess.data.fen_codec import fen_to_board_state
from modalchess.data.preprocessing_common import iter_records_from_path, write_yaml
from modalchess.data.tensor_codec import encode_fen_history
from modalchess.train.train_spatial_baseline import build_model_from_config, count_model_parameters
from modalchess.utils.device import resolve_device


@dataclass(slots=True)
class EmbeddingExportConfig:
    """Embedding export configuration."""

    batch_size: int = 64
    include_square_tokens: bool = False


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device)


def _meta_feature_tensor(fen: str) -> torch.Tensor:
    board_state = fen_to_board_state(fen)
    return torch.tensor(
        [
            float(board_state.meta.halfmove_clock),
            float(board_state.meta.fullmove_number),
            float(board_state.meta.repetition_count),
        ],
        dtype=torch.float32,
    )


def _rows_from_path(path: str | Path) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    return [dict(row) for row in iter_records_from_path(path_obj)]


def export_embeddings_for_checkpoint(
    *,
    checkpoint_path: str | Path,
    dataset_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    config: EmbeddingExportConfig | None = None,
) -> dict[str, Any]:
    """Export pooled embeddings for a frozen backbone checkpoint."""
    export_config = config or EmbeddingExportConfig()
    device = resolve_device()
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model_config = checkpoint["resolved_model_config"]
    model = build_model_from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    parameter_counts = count_model_parameters(model)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "seed": checkpoint.get("seed"),
        "git_hash": checkpoint.get("git_hash", "unknown"),
        "resolved_model_config": model_config,
        "parameter_counts": parameter_counts,
        "datasets": {},
        "config": asdict(export_config),
    }

    history_length = int(model_config["history_length"])
    with torch.no_grad():
        for dataset_name, dataset_path in dataset_paths.items():
            rows = _rows_from_path(dataset_path)
            output_rows: list[dict[str, Any]] = []
            for start in range(0, len(rows), export_config.batch_size):
                batch_rows = rows[start : start + export_config.batch_size]
                if not batch_rows:
                    continue
                board_planes = torch.stack(
                    [
                        encode_fen_history(
                            row.get("history_fens") or [row["fen"]],
                            history_length=history_length,
                        )
                        for row in batch_rows
                    ],
                    dim=0,
                ).to(device)
                meta_features = torch.stack(
                    [_meta_feature_tensor(str(row["fen"])) for row in batch_rows],
                    dim=0,
                ).to(device)
                outputs = model(board_planes=board_planes, meta_features=meta_features)
                board_pooled = outputs["board_pooled"].detach().cpu()
                context_pooled = outputs["context_pooled"].detach().cpu()
                square_tokens = outputs["tokens"].detach().cpu() if export_config.include_square_tokens else None
                for index, row in enumerate(batch_rows):
                    payload = {
                        "position_id": row.get("matched_position_id") or row.get("position_id"),
                        "split": row.get("split"),
                        "source": row.get("source"),
                        "target_move_uci": row.get("target_move_uci") or row.get("matched_target_move_uci"),
                        "board_pooled": board_pooled[index].tolist(),
                        "context_pooled": context_pooled[index].tolist(),
                        "checkpoint_path": str(checkpoint_path),
                        "seed": checkpoint.get("seed"),
                        "git_hash": checkpoint.get("git_hash", "unknown"),
                        "model_parameter_count": parameter_counts["model_parameter_count"],
                        "sidecar_id": row.get("sidecar_id"),
                        "alignment_type": row.get("alignment_type"),
                    }
                    if square_tokens is not None:
                        payload["square_tokens"] = square_tokens[index].tolist()
                    output_rows.append(payload)

            output_path = destination / f"{dataset_name}_embeddings.jsonl"
            with output_path.open("w", encoding="utf-8") as handle:
                for row in output_rows:
                    handle.write(json.dumps(row) + "\n")
            manifest["datasets"][dataset_name] = {
                "input_path": str(dataset_path),
                "output_path": str(output_path),
                "row_count": len(output_rows),
            }

    manifest_path = destination / "embedding_manifest.yaml"
    write_yaml(manifest_path, manifest)
    return {
        "manifest_path": str(manifest_path),
        "datasets": manifest["datasets"],
    }
