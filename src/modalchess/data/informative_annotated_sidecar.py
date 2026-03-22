"""Build higher-information annotated comment subsets for evaluation-only retrieval."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from modalchess.data.comment_duplicate_audit import generate_comment_duplicate_audit
from modalchess.data.comment_informativeness import (
    CommentInformativenessConfig,
    annotate_informativeness_rows,
    generate_comment_informativeness_audit,
)
from modalchess.data.preprocessing_common import special_rule_flags, write_jsonl, write_yaml


@dataclass(slots=True)
class InformativeAnnotatedSidecarConfig:
    primary_variant: str = "medium_high_only"
    informativeness_config: CommentInformativenessConfig = field(default_factory=CommentInformativenessConfig)
    salt: str = "modalchess_week14_informative_sidecar"


def _variant_root(output_root: Path, variant_name: str) -> Path:
    return output_root / "variants" / variant_name


def _copy_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _enriched_row(row: dict[str, Any], *, variant_name: str) -> dict[str, Any]:
    payload = dict(row)
    info = dict(row["comment_informativeness"])
    payload["informativeness_score"] = float(info["informativeness_score"])
    payload["informativeness_bucket"] = str(info["informativeness_bucket"])
    payload["informative_variant"] = variant_name
    return payload


def _drop_payload(
    row: dict[str, Any],
    *,
    variant_name: str,
    drop_reason: str,
    threshold_used: float | None = None,
) -> dict[str, Any]:
    info = dict(row["comment_informativeness"])
    return {
        "variant": variant_name,
        "split": row.get("split"),
        "original_sidecar_id": row.get("sidecar_id"),
        "position_id": row.get("position_id"),
        "game_id": row.get("game_id"),
        "comment_source": row.get("comment_source"),
        "drop_reason": drop_reason,
        "informativeness_score": info["informativeness_score"],
        "informativeness_bucket": info["informativeness_bucket"],
        "threshold_used": threshold_used,
        "comment_text": row.get("comment_text"),
    }


def _source_quantile_threshold(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    config: InformativeAnnotatedSidecarConfig,
) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    quantile = config.informativeness_config.medium_source_quantile
    for split_name, rows in rows_by_split.items():
        scores_by_source: dict[str, list[float]] = {}
        for row in rows:
            source_name = str(row.get("comment_source") or "unknown")
            scores_by_source.setdefault(source_name, []).append(float(row["comment_informativeness"]["informativeness_score"]))
        split_thresholds: dict[str, float] = {}
        for source_name, values in scores_by_source.items():
            values = sorted(values)
            if not values:
                split_thresholds[source_name] = config.informativeness_config.medium_threshold
                continue
            rank = int(round((len(values) - 1) * quantile))
            split_thresholds[source_name] = max(config.informativeness_config.medium_threshold, float(values[rank]))
        thresholds[split_name] = split_thresholds
    return thresholds


def build_informative_annotated_sidecar(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_v2_clean",
    output_root: str | Path = "data/pilot/annotated_sidecar_v3_informative",
    config: InformativeAnnotatedSidecarConfig | None = None,
) -> dict[str, Any]:
    informative_config = config or InformativeAnnotatedSidecarConfig()
    input_path = Path(input_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "manifests").mkdir(parents=True, exist_ok=True)
    (output_path / "reports").mkdir(parents=True, exist_ok=True)
    (output_path / "variants").mkdir(parents=True, exist_ok=True)

    rows_by_split = annotate_informativeness_rows(
        input_root=input_path,
        config=informative_config.informativeness_config,
    )
    balanced_thresholds = _source_quantile_threshold(rows_by_split, config=informative_config)
    variant_names = (
        "high_informativeness_threshold",
        "balanced_informativeness_by_comment_source",
        "medium_high_only",
        "special_rule_priority_variant",
    )
    variant_summaries: dict[str, Any] = {}

    for variant_name in variant_names:
        variant_root = _variant_root(output_path, variant_name)
        variant_root.mkdir(parents=True, exist_ok=True)
        drop_manifest_path = output_path / "manifests" / f"{variant_name}_dropped_rows.jsonl"
        dropped_rows: list[dict[str, Any]] = []
        kept_counts: dict[str, int] = {}
        bucket_counts_by_split: dict[str, Counter[str]] = {}
        drop_reason_counts: Counter[str] = Counter()

        for split_name, rows in rows_by_split.items():
            kept_rows: list[dict[str, Any]] = []
            bucket_counts: Counter[str] = Counter()
            for row in rows:
                info = row["comment_informativeness"]
                score = float(info["informativeness_score"])
                bucket = str(info["informativeness_bucket"])
                keep = False
                drop_reason = "below_threshold"
                threshold_used: float | None = None
                if variant_name == "high_informativeness_threshold":
                    threshold_used = informative_config.informativeness_config.high_threshold
                    keep = score >= threshold_used
                    drop_reason = "below_high_threshold"
                elif variant_name == "medium_high_only":
                    threshold_used = informative_config.informativeness_config.medium_threshold
                    keep = score >= threshold_used
                    drop_reason = "below_medium_threshold"
                elif variant_name == "balanced_informativeness_by_comment_source":
                    source_name = str(row.get("comment_source") or "unknown")
                    threshold_used = balanced_thresholds[split_name].get(
                        source_name,
                        informative_config.informativeness_config.medium_threshold,
                    )
                    keep = score >= threshold_used
                    drop_reason = "below_source_quantile_threshold"
                elif variant_name == "special_rule_priority_variant":
                    threshold_used = informative_config.informativeness_config.medium_threshold
                    flags = special_rule_flags(str(row["fen"]), str(row["target_move_uci"]))
                    keep = score >= threshold_used or (
                        any(flags.values()) and score >= informative_config.informativeness_config.special_rule_floor
                    )
                    drop_reason = "below_special_rule_priority_threshold"
                else:
                    raise ValueError(f"unsupported informative variant: {variant_name}")

                if keep:
                    enriched = _enriched_row(row, variant_name=variant_name)
                    kept_rows.append(enriched)
                    bucket_counts[bucket] += 1
                else:
                    drop_reason_counts[drop_reason] += 1
                    dropped_rows.append(
                        _drop_payload(
                            row,
                            variant_name=variant_name,
                            drop_reason=drop_reason,
                            threshold_used=threshold_used,
                        )
                    )

            kept_counts[split_name] = write_jsonl(variant_root / f"{split_name}.jsonl", kept_rows)
            bucket_counts_by_split[split_name] = bucket_counts

        write_jsonl(drop_manifest_path, dropped_rows)
        variant_summaries[variant_name] = {
            "variant_root": str(variant_root),
            "drop_manifest_path": str(drop_manifest_path),
            "before_counts_by_split": {split_name: len(rows) for split_name, rows in rows_by_split.items()},
            "after_counts_by_split": kept_counts,
            "before_total_rows": sum(len(rows) for rows in rows_by_split.values()),
            "after_total_rows": sum(kept_counts.values()),
            "drop_reason_counts": dict(drop_reason_counts),
            "bucket_counts_by_split": {split_name: dict(counter) for split_name, counter in bucket_counts_by_split.items()},
        }

    primary_variant = informative_config.primary_variant
    primary_root = Path(str(variant_summaries[primary_variant]["variant_root"]))
    for split_name in ("train", "val", "test"):
        write_jsonl(output_path / f"{split_name}.jsonl", _copy_jsonl(primary_root / f"{split_name}.jsonl"))

    manifest = {
        "input_root": str(input_path),
        "primary_variant": primary_variant,
        "config": {
            "primary_variant": informative_config.primary_variant,
            "informativeness_config": asdict(informative_config.informativeness_config),
            "salt": informative_config.salt,
        },
        "variants": variant_summaries,
        "outputs": {
            split_name: str(output_path / f"{split_name}.jsonl")
            for split_name in ("train", "val", "test")
        },
    }
    manifest_path = output_path / "manifests" / "informative_sidecar_manifest.yaml"
    write_yaml(manifest_path, manifest)

    before_audit = generate_comment_informativeness_audit(
        input_root=input_path,
        config=informative_config.informativeness_config,
    )
    after_audit = generate_comment_informativeness_audit(
        input_root=output_path,
        config=informative_config.informativeness_config,
    )
    before_duplicate = generate_comment_duplicate_audit(input_root=input_path)
    after_duplicate = generate_comment_duplicate_audit(input_root=output_path)
    diff_payload = {
        "before_total_rows": before_audit["total_rows"],
        "after_total_rows": after_audit["total_rows"],
        "row_delta": after_audit["total_rows"] - before_audit["total_rows"],
        "before_score_distribution_by_split": before_audit["score_distribution_by_split"],
        "after_score_distribution_by_split": after_audit["score_distribution_by_split"],
        "before_exact_duplicate_share": before_duplicate["exact_comment_text"]["duplicate_row_share"],
        "after_exact_duplicate_share": after_duplicate["exact_comment_text"]["duplicate_row_share"],
        "before_cross_position_duplicate_share": before_duplicate["duplicate_comment_text_across_positions"]["duplicate_row_share"],
        "after_cross_position_duplicate_share": after_duplicate["duplicate_comment_text_across_positions"]["duplicate_row_share"],
    }
    report = {
        "input_root": str(input_path),
        "primary_variant": primary_variant,
        "variants": variant_summaries,
        "before_informativeness_audit": before_audit,
        "after_informativeness_audit": after_audit,
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
    report_json_path = output_path / "reports" / "informative_sidecar_report.json"
    report_md_path = output_path / "reports" / "informative_sidecar_report.md"
    diff_json_path = output_path / "reports" / "v2_clean_vs_v3_informative_diff.json"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    diff_json_path.write_text(json.dumps(diff_payload, indent=2), encoding="utf-8")

    lines = ["# Informative Annotated Sidecar Report", ""]
    lines.append(f"- primary_variant: `{primary_variant}`")
    lines.append(
        f"- rows: before={diff_payload['before_total_rows']}, after={diff_payload['after_total_rows']}, delta={diff_payload['row_delta']}"
    )
    lines.append(
        f"- exact_duplicate_share: before={diff_payload['before_exact_duplicate_share']:.6f}, after={diff_payload['after_exact_duplicate_share']:.6f}"
    )
    lines.append(
        f"- cross_position_duplicate_share: before={diff_payload['before_cross_position_duplicate_share']:.6f}, after={diff_payload['after_cross_position_duplicate_share']:.6f}"
    )
    for variant_name, payload in variant_summaries.items():
        lines.append("")
        lines.append(f"## {variant_name}")
        lines.append(f"- before_total_rows={payload['before_total_rows']}, after_total_rows={payload['after_total_rows']}")
        lines.append(f"- drop_reason_counts={payload['drop_reason_counts']}")
        lines.append(f"- bucket_counts_by_split={payload['bucket_counts_by_split']}")
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "manifest_path": str(manifest_path),
        "report_json": str(report_json_path),
        "report_md": str(report_md_path),
        "diff_json": str(diff_json_path),
        "primary_root": str(output_path),
        "variants": variant_summaries,
    }
