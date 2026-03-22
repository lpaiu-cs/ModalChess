"""Dedup-aware evaluation corpora for annotated comment retrieval."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from modalchess.data.comment_duplicate_audit import normalize_comment_text
from modalchess.data.preprocessing_common import iter_records_from_path, write_jsonl, write_yaml


@dataclass(slots=True)
class DedupCommentEvalConfig:
    primary_variant: str = "normalized_comment_dedup"
    normalized_mode: str = "nag_stripped"
    capped_duplicates_per_cluster: int = 3
    split_member_lists: bool = True
    salt: str = "modalchess_week12_dedup_eval"


def _load_rows_by_split(root: str | Path) -> dict[str, list[dict[str, Any]]]:
    input_root = Path(root)
    return {
        split_name: [dict(row) for row in iter_records_from_path(input_root / f"{split_name}.jsonl")]
        for split_name in ("train", "val", "test")
    }


def _cluster_id(split_name: str, variant_name: str, cluster_key: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{variant_name}:{split_name}:{cluster_key}".encode("utf-8")).hexdigest()[:16]
    return f"dedup_{digest}"


def _row_hash(row: dict[str, Any], salt: str) -> str:
    sidecar_id = str(row.get("sidecar_id") or row.get("probe_id") or "")
    return hashlib.sha256(f"{salt}:{sidecar_id}".encode("utf-8")).hexdigest()


def _group_rows(
    rows: list[dict[str, Any]],
    *,
    variant_name: str,
    config: DedupCommentEvalConfig,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        comment_text = str(row.get("comment_text") or "").strip()
        comment_source = str(row.get("comment_source") or "unknown")
        if not comment_text:
            cluster_key = f"empty::{row.get('sidecar_id') or row.get('probe_id')}"
        elif variant_name == "exact_comment_dedup":
            cluster_key = f"{comment_source}::{comment_text}"
        elif variant_name in {"normalized_comment_dedup", "capped_duplicates"}:
            normalized = normalize_comment_text(comment_text, mode=config.normalized_mode)
            cluster_key = f"{comment_source}::{normalized or comment_text}"
        else:
            raise ValueError(f"unsupported dedup variant: {variant_name}")
        groups[cluster_key].append(row)
    return groups


def _select_rows_from_groups(
    groups: dict[str, list[dict[str, Any]]],
    *,
    split_name: str,
    variant_name: str,
    config: DedupCommentEvalConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_rows: list[dict[str, Any]] = []
    cluster_records: list[dict[str, Any]] = []
    for cluster_key, cluster_rows in sorted(groups.items()):
        ranked_rows = sorted(cluster_rows, key=lambda row: _row_hash(row, config.salt))
        if variant_name == "capped_duplicates":
            kept_rows = ranked_rows[: config.capped_duplicates_per_cluster]
        else:
            kept_rows = ranked_rows[:1]
        cluster_id = _cluster_id(split_name, variant_name, cluster_key, config.salt)
        member_sidecar_ids = [str(row.get("sidecar_id") or row.get("probe_id") or "") for row in ranked_rows]
        cluster_record = {
            "dedup_cluster_id": cluster_id,
            "split": split_name,
            "variant": variant_name,
            "cluster_key": cluster_key,
            "cluster_size": len(ranked_rows),
            "kept_size": len(kept_rows),
            "comment_source": str(ranked_rows[0].get("comment_source") or "unknown"),
        }
        if config.split_member_lists:
            cluster_record["member_sidecar_ids"] = member_sidecar_ids
        cluster_records.append(cluster_record)
        for keep_index, row in enumerate(kept_rows):
            enriched_row = dict(row)
            enriched_row["dedup_cluster_id"] = cluster_id
            enriched_row["dedup_variant"] = variant_name
            enriched_row["dedup_cluster_size"] = len(ranked_rows)
            enriched_row["dedup_cluster_keep_rank"] = keep_index
            enriched_row["dedup_normalization_mode"] = config.normalized_mode
            selected_rows.append(enriched_row)
    selected_rows.sort(key=lambda row: _row_hash(row, config.salt))
    return selected_rows, cluster_records


def _comment_source_proportions(rows: list[dict[str, Any]]) -> dict[str, float]:
    counter: Counter[str] = Counter(str(row.get("comment_source") or "unknown") for row in rows)
    total = sum(counter.values())
    return {
        key: (count / total) if total else 0.0
        for key, count in sorted(counter.items())
    }


def build_dedup_comment_eval(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_v1",
    output_root: str | Path = "data/pilot/annotated_sidecar_eval_v2",
    config: DedupCommentEvalConfig | None = None,
) -> dict[str, Any]:
    dedup_config = config or DedupCommentEvalConfig()
    input_path = Path(input_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_path / "manifests"
    reports_dir = output_path / "reports"
    variants_dir = output_path / "variants"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    variants_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = _load_rows_by_split(input_path)
    variant_names = ["exact_comment_dedup", "normalized_comment_dedup", "capped_duplicates"]
    variant_summaries: dict[str, Any] = {}

    for variant_name in variant_names:
        variant_root = variants_dir / variant_name
        variant_root.mkdir(parents=True, exist_ok=True)
        cluster_map_path = manifests_dir / f"{variant_name}_cluster_members.jsonl"
        cluster_map_records: list[dict[str, Any]] = []
        split_counts: dict[str, int] = {}
        before_counts: dict[str, int] = {}
        source_props: dict[str, dict[str, float]] = {}

        for split_name, split_rows in rows_by_split.items():
            before_counts[split_name] = len(split_rows)
            groups = _group_rows(split_rows, variant_name=variant_name, config=dedup_config)
            selected_rows, cluster_records = _select_rows_from_groups(
                groups,
                split_name=split_name,
                variant_name=variant_name,
                config=dedup_config,
            )
            split_counts[split_name] = write_jsonl(variant_root / f"{split_name}.jsonl", selected_rows)
            cluster_map_records.extend(cluster_records)
            source_props[split_name] = _comment_source_proportions(selected_rows)

        write_jsonl(cluster_map_path, cluster_map_records)
        variant_summaries[variant_name] = {
            "variant_root": str(variant_root),
            "cluster_map_path": str(cluster_map_path),
            "before_counts_by_split": before_counts,
            "after_counts_by_split": split_counts,
            "before_total_rows": sum(before_counts.values()),
            "after_total_rows": sum(split_counts.values()),
            "cluster_count": len(cluster_map_records),
            "multi_member_cluster_count": sum(int(record["cluster_size"] > 1) for record in cluster_map_records),
            "comment_source_proportions_by_split": source_props,
            "dedup_rule": (
                "cluster by comment_source + exact comment_text"
                if variant_name == "exact_comment_dedup"
                else (
                    f"cluster by comment_source + normalized comment_text ({dedup_config.normalized_mode})"
                    if variant_name == "normalized_comment_dedup"
                    else (
                        "cluster by comment_source + normalized comment_text "
                        f"({dedup_config.normalized_mode}); keep first {dedup_config.capped_duplicates_per_cluster}"
                    )
                )
            ),
        }

    primary_root = Path(str(variant_summaries[dedup_config.primary_variant]["variant_root"]))
    for split_name in ("train", "val", "test"):
        write_jsonl(
            output_path / f"{split_name}.jsonl",
            [dict(row) for row in iter_records_from_path(primary_root / f"{split_name}.jsonl")],
        )

    manifest = {
        "input_root": str(input_path),
        "primary_variant": dedup_config.primary_variant,
        "config": asdict(dedup_config),
        "variants": variant_summaries,
        "outputs": {
            split_name: str(output_path / f"{split_name}.jsonl")
            for split_name in ("train", "val", "test")
        },
    }
    manifest_path = manifests_dir / "dedup_eval_manifest.yaml"
    write_yaml(manifest_path, manifest)

    report = {
        "input_root": str(input_path),
        "primary_variant": dedup_config.primary_variant,
        "variants": variant_summaries,
    }
    report_json_path = reports_dir / "dedup_eval_report.json"
    report_md_path = reports_dir / "dedup_eval_report.md"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Dedup Eval Report", ""]
    lines.append(f"- primary_variant: `{dedup_config.primary_variant}`")
    for variant_name, payload in variant_summaries.items():
        lines.append("")
        lines.append(f"## {variant_name}")
        lines.append(
            f"- before_total_rows={payload['before_total_rows']}, after_total_rows={payload['after_total_rows']}, "
            f"cluster_count={payload['cluster_count']}, multi_member_cluster_count={payload['multi_member_cluster_count']}"
        )
        lines.append(f"- dedup_rule: {payload['dedup_rule']}")
        for split_name in ("train", "val", "test"):
            lines.append(
                f"- `{split_name}`: before={payload['before_counts_by_split'][split_name]}, "
                f"after={payload['after_counts_by_split'][split_name]}"
            )
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "manifest_path": str(manifest_path),
        "report_json": str(report_json_path),
        "report_md": str(report_md_path),
        "primary_root": str(output_path),
        "variant_summaries": variant_summaries,
    }


def build_comment_retrieval_eval_regime_v2(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_eval_v2",
    output_root: str | Path = "outputs/week12/comment_retrieval_v2",
    config,
) -> dict[str, Any]:
    from modalchess.data.comment_retrieval_eval import build_comment_retrieval_eval_regime

    input_path = Path(input_root)
    result = build_comment_retrieval_eval_regime(
        input_root=input_path,
        output_root=output_root,
        config=config,
    )
    manifest_path = Path(result["manifest_path"])
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    dedup_manifest_path = input_path / "manifests" / "dedup_eval_manifest.yaml"
    dedup_manifest = {}
    if dedup_manifest_path.exists():
        dedup_manifest = yaml.safe_load(dedup_manifest_path.read_text(encoding="utf-8")) or {}
    manifest["corpus_mode"] = "dedup_aware"
    manifest["dedup_manifest_path"] = str(dedup_manifest_path)
    manifest["dedup_mode"] = dedup_manifest.get("primary_variant")
    manifest["subset_mode"] = manifest.get("evaluation_mode")
    manifest["source_input_root"] = dedup_manifest.get("input_root")
    write_yaml(manifest_path, manifest)
    result["manifest"] = manifest
    return result
