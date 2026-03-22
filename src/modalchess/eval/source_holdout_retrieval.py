"""Source-holdout and source-type retrieval evaluation on frozen embeddings."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

import yaml
import torch

from modalchess.data.preprocessing_common import iter_records_from_path, write_yaml
from modalchess.eval.raw_text_retrieval import run_raw_text_retrieval_probes


ROW_ALIGNED_LIST_KEYS = ("probe_id", "position_id", "source", "split", "fen", "target_move_uci")
ROW_ALIGNED_TENSOR_KEYS = ("board_pooled", "context_pooled", "square_tokens")


def _load_regime_rows(regime_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        split_name: [
            dict(row)
            for row in iter_records_from_path(regime_dir / f"annotated_sidecar_{split_name}.jsonl")
        ]
        for split_name in ("train", "val", "test")
    }


def _probe_ids(rows: Iterable[dict[str, Any]]) -> list[str]:
    return [str(row.get("probe_id") or row.get("sidecar_id") or "") for row in rows]


def _filter_embedding_payload(payload: dict[str, Any], probe_ids: list[str]) -> dict[str, Any]:
    index_by_probe_id = {
        str(probe_id): index
        for index, probe_id in enumerate(payload.get("probe_id") or [])
    }
    indices = [index_by_probe_id[probe_id] for probe_id in probe_ids]
    index_tensor = torch.tensor(indices, dtype=torch.long)
    filtered: dict[str, Any] = {}
    for key, value in payload.items():
        if key in ROW_ALIGNED_LIST_KEYS:
            filtered[key] = [value[index] for index in indices]
        elif key in ROW_ALIGNED_TENSOR_KEYS and isinstance(value, torch.Tensor):
            filtered[key] = value.index_select(0, index_tensor)
        else:
            filtered[key] = value
    return filtered


def _prepare_regime_embedding_root(
    *,
    full_embedding_root: Path,
    regime_dir: Path,
    cache_root: Path,
    backbone_seeds: list[int],
) -> Path:
    regime_rows = _load_regime_rows(regime_dir)
    regime_embedding_root = cache_root / regime_dir.name
    regime_embedding_root.mkdir(parents=True, exist_ok=True)
    for backbone_name in ("g1", "g3"):
        for seed in backbone_seeds:
            source_dir = full_embedding_root / backbone_name / f"seed{seed}"
            destination_dir = regime_embedding_root / backbone_name / f"seed{seed}"
            destination_dir.mkdir(parents=True, exist_ok=True)
            manifest_payload = None
            for split_name in ("train", "val", "test"):
                payload = torch.load(
                    source_dir / f"annotated_sidecar_{split_name}_embeddings.pt",
                    map_location="cpu",
                )
                filtered_payload = _filter_embedding_payload(payload, _probe_ids(regime_rows[split_name]))
                torch.save(filtered_payload, destination_dir / f"annotated_sidecar_{split_name}_embeddings.pt")
                if manifest_payload is None:
                    manifest_payload = {
                        "checkpoint_path": filtered_payload.get("checkpoint_path"),
                        "seed": filtered_payload.get("seed"),
                        "git_hash": filtered_payload.get("git_hash"),
                        "model_parameter_count": filtered_payload.get("model_parameter_count"),
                    }
            if manifest_payload is not None:
                manifest_payload["datasets"] = {
                    split_name: {
                        "input_path": str(regime_dir / f"annotated_sidecar_{split_name}.jsonl"),
                        "output_path": str(destination_dir / f"annotated_sidecar_{split_name}_embeddings.pt"),
                        "row_count": len(regime_rows[split_name]),
                    }
                    for split_name in ("train", "val", "test")
                }
                write_yaml(destination_dir / "embedding_manifest.yaml", manifest_payload)
    return regime_embedding_root


def _select_regimes(manifest: dict[str, Any], categories: set[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for regime in manifest.get("regimes", []):
        if str(regime.get("category")) in categories:
            selected.append(dict(regime))
    for source_type, payload in (manifest.get("source_types") or {}).items():
        category = str(payload.get("category") or "source_type_ablation")
        if category in categories:
            selected.append(dict(payload))
    selected.sort(key=lambda item: (str(item.get("category")), str(item.get("regime_name"))))
    return selected


def _best_row(rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    return max(rows, key=lambda row: (float(row[metric_name]), -float(row["test_rows_mean"]), str(row["backbone"])))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(category: str, summary_rows: list[dict[str, Any]]) -> str:
    title = "# Source Holdout Retrieval Summary" if category == "source_holdout_retrieval" else "# Source Type Ablation Summary"
    lines = [title, ""]
    mixed_rows = [row for row in summary_rows if row["regime_name"] in {"mixed_baseline", "source_type__mixed"}]
    mixed_board = _best_row(mixed_rows, "strict_board_to_text_mrr_mean") if mixed_rows else None
    mixed_text = _best_row(mixed_rows, "strict_text_to_board_mrr_mean") if mixed_rows else None
    if mixed_board is not None and mixed_text is not None:
        lines.append(
            f"- mixed baseline strict board_to_text MRR: {mixed_board['strict_board_to_text_mrr_mean']:.4f} "
            f"({mixed_board['backbone']} / {mixed_board['pool']} / {mixed_board['probe_model']})"
        )
        lines.append(
            f"- mixed baseline strict text_to_board MRR: {mixed_text['strict_text_to_board_mrr_mean']:.4f} "
            f"({mixed_text['backbone']} / {mixed_text['pool']} / {mixed_text['probe_model']})"
        )
        lines.append("")
    by_regime: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_regime.setdefault(str(row["regime_name"]), []).append(row)
    for regime_name, regime_rows in sorted(by_regime.items()):
        best_board = _best_row(regime_rows, "strict_board_to_text_mrr_mean")
        best_text = _best_row(regime_rows, "strict_text_to_board_mrr_mean")
        random_r1 = 1.0 / max(int(best_board["test_rows_mean"]), 1)
        lines.append(
            f"- `{regime_name}`: strict_board_to_text_MRR={best_board['strict_board_to_text_mrr_mean']:.4f}, "
            f"strict_text_to_board_MRR={best_text['strict_text_to_board_mrr_mean']:.4f}, "
            f"random_R@1~{random_r1:.6f}"
        )
    return "\n".join(lines) + "\n"


def run_source_holdout_retrieval(
    *,
    holdout_root: str | Path = "data/pilot/annotated_sidecar_holdout_v1",
    full_embedding_root: str | Path = "outputs/week16/full_eval_embeddings",
    output_dir: str | Path = "outputs/week16/source_holdout_retrieval",
    categories: set[str] | None = None,
    backbone_seeds: list[int] | None = None,
    mate_min_df: int = 25,
    max_vocab_size: int = 512,
    probe_models: list[str] | None = None,
) -> dict[str, Any]:
    category_set = categories or {"mixed_baseline", "coarse_source_holdout", "source_family_holdout"}
    seed_list = backbone_seeds or [11, 17, 23]
    holdout_path = Path(holdout_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = holdout_path / "manifests" / "holdout_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    selected_regimes = _select_regimes(manifest, category_set)

    cache_root = output_path / "embedding_cache"
    per_regime_results: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    regime_summaries: list[dict[str, Any]] = []

    for regime in selected_regimes:
        regime_dir = Path(str(regime["regime_dir"]))
        regime_embedding_root = _prepare_regime_embedding_root(
            full_embedding_root=Path(full_embedding_root),
            regime_dir=regime_dir,
            cache_root=cache_root,
            backbone_seeds=seed_list,
        )
        result = run_raw_text_retrieval_probes(
            embedding_root=regime_embedding_root,
            corpus_root=regime_dir,
            output_dir=output_path / "per_regime" / str(regime["regime_name"]),
            backbone_seeds=seed_list,
            mate_min_df=mate_min_df,
            puzzle_min_df=mate_min_df,
            max_vocab_size=max_vocab_size,
            families=["annotated_sidecar"],
            probe_models=probe_models or ["linear"],
            output_prefix="comment_retrieval",
        )
        for row in result["results"]:
            per_regime_results.append({**regime, **row})
        for row in result["aggregate"]:
            enriched = {**regime, **row}
            aggregate_rows.append(enriched)
        regime_rows = [row for row in aggregate_rows if row["regime_name"] == regime["regime_name"]]
        regime_summaries.append(
            {
                **regime,
                "best_strict_board_to_text": _best_row(regime_rows, "strict_board_to_text_mrr_mean"),
                "best_strict_text_to_board": _best_row(regime_rows, "strict_text_to_board_mrr_mean"),
            }
        )

    results_json_path = output_path / "results.json"
    results_csv_path = output_path / "results.csv"
    summary_path = output_path / "summary.md"
    payload = {
        "results": per_regime_results,
        "aggregate": aggregate_rows,
        "regime_summaries": regime_summaries,
    }
    results_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(results_csv_path, aggregate_rows)
    summary_path.write_text(
        _summary_markdown(output_path.name, aggregate_rows),
        encoding="utf-8",
    )
    return {
        "results_json": str(results_json_path),
        "results_csv": str(results_csv_path),
        "summary_path": str(summary_path),
        "payload": payload,
    }
