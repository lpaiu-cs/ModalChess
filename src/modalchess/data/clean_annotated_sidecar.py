"""Build cleaner move-conditioned annotated comment corpora with provenance-preserving filtering."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from modalchess.data.comment_boilerplate_audit import (
    CommentBoilerplateConfig,
    annotate_comment_rows,
    generate_comment_boilerplate_audit,
)
from modalchess.data.comment_duplicate_audit import generate_comment_duplicate_audit
from modalchess.data.preprocessing_common import iter_records_from_path, write_jsonl, write_yaml


BASE_DROP_CATEGORIES = {
    "markup_only",
    "pgn_markup_heavy",
    "result_comment",
    "symbol_only",
    "engine_template",
    "short_repeated_template",
    "low_lexical_diversity_repeated",
}


@dataclass(slots=True)
class CleanAnnotatedSidecarConfig:
    primary_variant: str = "keep_comment_source_balance"
    boilerplate_config: CommentBoilerplateConfig = field(default_factory=CommentBoilerplateConfig)
    template_cap_per_cluster: int = 3
    template_cap_per_source_cluster: int = 3
    salt: str = "modalchess_week13_clean_sidecar"


def _stable_rank_key(sidecar_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{sidecar_id}".encode("utf-8")).hexdigest()


def _copy_rows(path: str | Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_records_from_path(path)]


def _enriched_keep_row(
    row: dict[str, Any],
    *,
    variant_name: str,
) -> dict[str, Any]:
    analysis = dict(row["comment_boilerplate"])
    enriched = dict(row)
    enriched["original_comment_text"] = str(row.get("comment_text") or "")
    enriched["comment_text"] = str(analysis["plain_text"] or row.get("comment_text") or "").strip()
    enriched["clean_variant"] = variant_name
    enriched["boilerplate_categories"] = list(analysis["categories"])
    enriched["comment_template"] = str(analysis["normalized_template"])
    return enriched


def _drop_payload(
    row: dict[str, Any],
    *,
    variant_name: str,
    drop_reason: str,
    boilerplate_category: str | None = None,
) -> dict[str, Any]:
    analysis = dict(row["comment_boilerplate"])
    return {
        "variant": variant_name,
        "split": row.get("split"),
        "original_sidecar_id": row.get("sidecar_id"),
        "position_id": row.get("position_id"),
        "game_id": row.get("game_id"),
        "source_file": row.get("source_file"),
        "comment_source": row.get("comment_source"),
        "drop_reason": drop_reason,
        "boilerplate_category": boilerplate_category,
        "boilerplate_categories": list(analysis["categories"]),
        "comment_text": row.get("comment_text"),
        "cleaned_comment_text": analysis["plain_text"],
        "normalized_template": analysis["normalized_template"],
    }


def _variant_drop_categories(variant_name: str) -> set[str]:
    if variant_name == "remove_eval_markup_heavy":
        return {"markup_only", "pgn_markup_heavy"}
    if variant_name in {"remove_short_boilerplate", "keep_comment_source_balance", "template_capped"}:
        return set(BASE_DROP_CATEGORIES)
    raise ValueError(f"unsupported clean variant: {variant_name}")


def _cap_rows_by_cluster(
    candidate_rows: list[dict[str, Any]],
    *,
    variant_name: str,
    config: CleanAnnotatedSidecarConfig,
    split_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if variant_name not in {"keep_comment_source_balance", "template_capped"}:
        return candidate_rows, []

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        analysis = row["comment_boilerplate"]
        template = str(analysis["normalized_template"] or row.get("sidecar_id"))
        if variant_name == "keep_comment_source_balance":
            cluster_key = f"{row.get('comment_source') or 'unknown'}::{template}"
            cap = config.template_cap_per_source_cluster
        else:
            cluster_key = template
            cap = config.template_cap_per_cluster
        grouped[cluster_key].append(row)

    kept_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    for _cluster_key, cluster_rows in sorted(grouped.items()):
        ranked_rows = sorted(
            cluster_rows,
            key=lambda row: _stable_rank_key(str(row.get("sidecar_id") or row.get("probe_id") or ""), f"{config.salt}:{split_name}:{variant_name}"),
        )
        kept_rows.extend(ranked_rows[:cap])
        for dropped_row in ranked_rows[cap:]:
            dropped_rows.append(
                _drop_payload(
                    dropped_row,
                    variant_name=variant_name,
                    drop_reason="template_cap",
                )
            )
    kept_rows.sort(
        key=lambda row: _stable_rank_key(str(row.get("sidecar_id") or row.get("probe_id") or ""), f"{config.salt}:{split_name}:{variant_name}:final")
    )
    return kept_rows, dropped_rows


def _variant_root(output_root: Path, variant_name: str) -> Path:
    return output_root / "variants" / variant_name


def build_clean_annotated_sidecar(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_v1",
    output_root: str | Path = "data/pilot/annotated_sidecar_v2_clean",
    config: CleanAnnotatedSidecarConfig | None = None,
) -> dict[str, Any]:
    clean_config = config or CleanAnnotatedSidecarConfig()
    input_path = Path(input_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "manifests").mkdir(parents=True, exist_ok=True)
    (output_path / "reports").mkdir(parents=True, exist_ok=True)
    (output_path / "variants").mkdir(parents=True, exist_ok=True)

    rows_by_split = annotate_comment_rows(input_root=input_path, config=clean_config.boilerplate_config)
    variant_names = (
        "remove_eval_markup_heavy",
        "remove_short_boilerplate",
        "keep_comment_source_balance",
        "template_capped",
    )
    variant_summaries: dict[str, Any] = {}

    for variant_name in variant_names:
        variant_root = _variant_root(output_path, variant_name)
        variant_root.mkdir(parents=True, exist_ok=True)
        kept_split_counts: dict[str, int] = {}
        drop_reason_counts: Counter[str] = Counter()
        boilerplate_drop_counts: Counter[str] = Counter()
        source_proportions_by_split: dict[str, dict[str, float]] = {}
        dropped_rows_manifest_path = output_path / "manifests" / f"{variant_name}_dropped_rows.jsonl"
        dropped_rows_buffer: list[dict[str, Any]] = []

        for split_name, split_rows in rows_by_split.items():
            candidate_rows: list[dict[str, Any]] = []
            for row in split_rows:
                categories = list(row["comment_boilerplate"]["categories"])
                plain_text = str(row["comment_boilerplate"]["plain_text"] or "").strip()
                drop_category = next((category for category in categories if category in _variant_drop_categories(variant_name)), None)
                if drop_category is not None:
                    drop_reason_counts[f"boilerplate:{drop_category}"] += 1
                    boilerplate_drop_counts[drop_category] += 1
                    dropped_rows_buffer.append(
                        _drop_payload(
                            row,
                            variant_name=variant_name,
                            drop_reason=f"boilerplate:{drop_category}",
                            boilerplate_category=drop_category,
                        )
                    )
                    continue
                if not plain_text:
                    drop_reason_counts["empty_after_cleanup"] += 1
                    dropped_rows_buffer.append(
                        _drop_payload(
                            row,
                            variant_name=variant_name,
                            drop_reason="empty_after_cleanup",
                        )
                    )
                    continue
                candidate_rows.append(_enriched_keep_row(row, variant_name=variant_name))

            kept_rows, cap_drops = _cap_rows_by_cluster(
                candidate_rows,
                variant_name=variant_name,
                config=clean_config,
                split_name=split_name,
            )
            for drop_payload in cap_drops:
                drop_reason_counts[drop_payload["drop_reason"]] += 1
            dropped_rows_buffer.extend(cap_drops)
            kept_split_counts[split_name] = write_jsonl(variant_root / f"{split_name}.jsonl", kept_rows)
            source_counts = Counter(str(row.get("comment_source") or "unknown") for row in kept_rows)
            source_total = sum(source_counts.values())
            source_proportions_by_split[split_name] = {
                source_name: (count / source_total) if source_total else 0.0
                for source_name, count in sorted(source_counts.items())
            }

        write_jsonl(dropped_rows_manifest_path, dropped_rows_buffer)
        variant_summaries[variant_name] = {
            "variant_root": str(variant_root),
            "drop_manifest_path": str(dropped_rows_manifest_path),
            "before_counts_by_split": {split_name: len(rows) for split_name, rows in rows_by_split.items()},
            "after_counts_by_split": kept_split_counts,
            "before_total_rows": sum(len(rows) for rows in rows_by_split.values()),
            "after_total_rows": sum(kept_split_counts.values()),
            "drop_reason_counts": dict(drop_reason_counts),
            "boilerplate_drop_counts": dict(boilerplate_drop_counts),
            "comment_source_proportions_by_split": source_proportions_by_split,
        }

    primary_variant = clean_config.primary_variant
    primary_root = Path(str(variant_summaries[primary_variant]["variant_root"]))
    for split_name in ("train", "val", "test"):
        write_jsonl(output_path / f"{split_name}.jsonl", _copy_rows(primary_root / f"{split_name}.jsonl"))

    manifest = {
        "input_root": str(input_path),
        "primary_variant": primary_variant,
        "config": {
            "primary_variant": clean_config.primary_variant,
            "boilerplate_config": asdict(clean_config.boilerplate_config),
            "template_cap_per_cluster": clean_config.template_cap_per_cluster,
            "template_cap_per_source_cluster": clean_config.template_cap_per_source_cluster,
            "salt": clean_config.salt,
        },
        "variants": variant_summaries,
        "outputs": {
            split_name: str(output_path / f"{split_name}.jsonl")
            for split_name in ("train", "val", "test")
        },
    }
    manifest_path = output_path / "manifests" / "clean_sidecar_manifest.yaml"
    write_yaml(manifest_path, manifest)

    before_boilerplate = generate_comment_boilerplate_audit(input_root=input_path, config=clean_config.boilerplate_config)
    after_boilerplate = generate_comment_boilerplate_audit(input_root=output_path, config=clean_config.boilerplate_config)
    before_duplicate = generate_comment_duplicate_audit(input_root=input_path)
    after_duplicate = generate_comment_duplicate_audit(input_root=output_path)
    diff_payload = {
        "before_total_rows": before_boilerplate["total_rows"],
        "after_total_rows": after_boilerplate["total_rows"],
        "row_delta": after_boilerplate["total_rows"] - before_boilerplate["total_rows"],
        "before_exact_duplicate_share": before_duplicate["exact_comment_text"]["duplicate_row_share"],
        "after_exact_duplicate_share": after_duplicate["exact_comment_text"]["duplicate_row_share"],
        "before_cross_position_duplicate_share": before_duplicate["duplicate_comment_text_across_positions"]["duplicate_row_share"],
        "after_cross_position_duplicate_share": after_duplicate["duplicate_comment_text_across_positions"]["duplicate_row_share"],
        "before_boilerplate_counts": before_boilerplate["counts_by_boilerplate_category"],
        "after_boilerplate_counts": after_boilerplate["counts_by_boilerplate_category"],
    }

    report = {
        "input_root": str(input_path),
        "primary_variant": primary_variant,
        "variants": variant_summaries,
        "before_boilerplate_audit": before_boilerplate,
        "after_boilerplate_audit": after_boilerplate,
        "before_duplicate_summary": {
            "exact_comment_text": before_duplicate["exact_comment_text"],
            "duplicate_comment_text_across_positions": before_duplicate["duplicate_comment_text_across_positions"],
        },
        "after_duplicate_summary": {
            "exact_comment_text": after_duplicate["exact_comment_text"],
            "duplicate_comment_text_across_positions": after_duplicate["duplicate_comment_text_across_positions"],
        },
        "diff": diff_payload,
    }
    report_json_path = output_path / "reports" / "clean_sidecar_report.json"
    report_md_path = output_path / "reports" / "clean_sidecar_report.md"
    diff_json_path = output_path / "reports" / "v1_vs_v2_clean_diff.json"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    diff_json_path.write_text(json.dumps(diff_payload, indent=2), encoding="utf-8")

    lines = ["# Clean Annotated Sidecar Report", ""]
    lines.append(f"- primary_variant: `{primary_variant}`")
    lines.append(
        f"- rows: before={diff_payload['before_total_rows']}, after={diff_payload['after_total_rows']}, "
        f"delta={diff_payload['row_delta']}"
    )
    lines.append(
        f"- exact_duplicate_share: before={diff_payload['before_exact_duplicate_share']:.6f}, "
        f"after={diff_payload['after_exact_duplicate_share']:.6f}"
    )
    lines.append(
        f"- cross_position_duplicate_share: before={diff_payload['before_cross_position_duplicate_share']:.6f}, "
        f"after={diff_payload['after_cross_position_duplicate_share']:.6f}"
    )
    for variant_name, payload in variant_summaries.items():
        lines.append("")
        lines.append(f"## {variant_name}")
        lines.append(
            f"- before_total_rows={payload['before_total_rows']}, after_total_rows={payload['after_total_rows']}"
        )
        lines.append(f"- drop_reason_counts={payload['drop_reason_counts']}")
        lines.append(f"- boilerplate_drop_counts={payload['boilerplate_drop_counts']}")
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "manifest_path": str(manifest_path),
        "report_json": str(report_json_path),
        "report_md": str(report_md_path),
        "diff_json": str(diff_json_path),
        "primary_root": str(output_path),
        "variants": variant_summaries,
    }
