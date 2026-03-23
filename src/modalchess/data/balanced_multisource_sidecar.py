"""Build family-balanced and style-normalized multisource annotated comment corpora."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from modalchess.data.comment_boilerplate_audit import strip_pgn_markup
from modalchess.data.preprocessing_common import iter_records_from_path, write_jsonl, write_yaml


def _source_type(row: dict[str, Any]) -> str:
    source_name = str(row.get("source") or "")
    if source_name == "waterhorse_annotated_pgn":
        return "waterhorse_only"
    if source_name.startswith("mate_"):
        return "mate_only"
    return "other"


@dataclass(slots=True)
class BalancedMultisourceConfig:
    family_caps: dict[str, int] = field(default_factory=lambda: {"train": 5000, "val": 700, "test": 700})
    source_type_caps: dict[str, int] = field(default_factory=lambda: {"train": 25000, "val": 3500, "test": 3500})
    salt: str = "modalchess_week17_balanced"
    primary_variant: str = "family_balanced_plus_style_normalized"


def _load_rows_by_split(root: str | Path) -> dict[str, list[dict[str, Any]]]:
    input_root = Path(root)
    return {
        split_name: [dict(row) for row in iter_records_from_path(input_root / f"{split_name}.jsonl")]
        for split_name in ("train", "val", "test")
    }


def _stable_rank(row: dict[str, Any], *, split_name: str, variant_name: str, salt: str) -> str:
    row_id = str(row.get("sidecar_id") or row.get("probe_id") or row.get("position_id") or "")
    return hashlib.sha256(f"{salt}:{variant_name}:{split_name}:{row_id}".encode("utf-8")).hexdigest()


def _capped_rows(
    rows: list[dict[str, Any]],
    *,
    split_name: str,
    variant_name: str,
    salt: str,
    key_fn,
    cap: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    selected: list[dict[str, Any]] = []
    before_counts = {group_name: len(group_rows) for group_name, group_rows in sorted(groups.items())}
    after_counts: dict[str, int] = {}
    for group_name, group_rows in sorted(groups.items()):
        ranked_rows = sorted(group_rows, key=lambda row: _stable_rank(row, split_name=split_name, variant_name=variant_name, salt=salt))
        chosen_rows = ranked_rows[:cap] if cap > 0 else ranked_rows
        selected.extend(chosen_rows)
        after_counts[group_name] = len(chosen_rows)
    selected.sort(key=lambda row: _stable_rank(row, split_name=split_name, variant_name=f"{variant_name}:final", salt=salt))
    return selected, before_counts, after_counts


def _normalize_comment_text(row: dict[str, Any]) -> dict[str, Any]:
    original_text = str(row.get("original_comment_text") or row.get("comment_text") or "")
    normalized_text = strip_pgn_markup(original_text)
    payload = dict(row)
    payload["original_comment_text"] = original_text
    payload["normalized_comment_text"] = normalized_text
    payload["style_normalization_mode"] = "strip_pgn_markup"
    payload["style_normalization_applied"] = normalized_text != original_text
    if normalized_text:
        payload["comment_text"] = normalized_text
    return payload


def _variant_rows(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    variant_name: str,
    config: BalancedMultisourceConfig,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    variant_rows: dict[str, list[dict[str, Any]]] = {}
    cap_report: dict[str, Any] = {}
    for split_name, rows in rows_by_split.items():
        if variant_name in {"family_balanced", "family_balanced_plus_style_normalized"}:
            selected_rows, before_counts, after_counts = _capped_rows(
                rows,
                split_name=split_name,
                variant_name=variant_name,
                salt=config.salt,
                key_fn=lambda row: row.get("source_family") or "unknown",
                cap=int(config.family_caps.get(split_name, 0)),
            )
        elif variant_name == "source_type_balanced":
            selected_rows, before_counts, after_counts = _capped_rows(
                rows,
                split_name=split_name,
                variant_name=variant_name,
                salt=config.salt,
                key_fn=_source_type,
                cap=int(config.source_type_caps.get(split_name, 0)),
            )
        else:
            raise ValueError(f"unsupported balanced variant: {variant_name}")

        if variant_name == "family_balanced_plus_style_normalized":
            selected_rows = [_normalize_comment_text(row) for row in selected_rows]
        else:
            selected_rows = [dict(row) for row in selected_rows]
            for row in selected_rows:
                row.setdefault("original_comment_text", str(row.get("comment_text") or ""))
                row["normalized_comment_text"] = str(row.get("comment_text") or "")
                row["style_normalization_mode"] = "none"
                row["style_normalization_applied"] = False

        variant_rows[split_name] = selected_rows
        cap_report[split_name] = {
            "before_counts": before_counts,
            "after_counts": after_counts,
        }
    return variant_rows, cap_report


def _count_by(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field_name) or "unknown") for row in rows).most_common())


def _share_by(rows: list[dict[str, Any]], field_name: str) -> dict[str, float]:
    counter = Counter(str(row.get(field_name) or "unknown") for row in rows)
    total = sum(counter.values())
    return {
        key: (count / total) if total else 0.0
        for key, count in sorted(counter.items())
    }


def build_balanced_multisource_sidecar(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_v4_multisource",
    output_root: str | Path = "data/pilot/annotated_sidecar_v5_balanced",
    config: BalancedMultisourceConfig | None = None,
) -> dict[str, Any]:
    balanced_config = config or BalancedMultisourceConfig()
    input_path = Path(input_root)
    output_path = Path(output_root)
    reports_dir = output_path / "reports"
    manifests_dir = output_path / "manifests"
    variants_dir = output_path / "variants"
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    variants_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = _load_rows_by_split(input_path)
    all_input_rows = [row for rows in rows_by_split.values() for row in rows]
    input_family_counts = _count_by(all_input_rows, "source_family")
    input_family_share = (max(input_family_counts.values()) / len(all_input_rows)) if all_input_rows else 0.0

    variant_names = (
        "family_balanced",
        "source_type_balanced",
        "family_balanced_plus_style_normalized",
    )
    variant_reports: dict[str, Any] = {}
    diff_payload: dict[str, Any] = {
        "input_total_rows": len(all_input_rows),
        "input_family_count": len(input_family_counts),
        "input_largest_family_share": input_family_share,
        "variants": {},
    }

    for variant_name in variant_names:
        variant_root = variants_dir / variant_name
        variant_root.mkdir(parents=True, exist_ok=True)
        variant_rows_by_split, cap_report = _variant_rows(rows_by_split, variant_name=variant_name, config=balanced_config)
        split_counts = {
            split_name: write_jsonl(variant_root / f"{split_name}.jsonl", rows)
            for split_name, rows in variant_rows_by_split.items()
        }
        all_variant_rows = [row for rows in variant_rows_by_split.values() for row in rows]
        source_family_counts = _count_by(all_variant_rows, "source_family")
        source_type_counts = _count_by(
            [{**row, "source_type": _source_type(row)} for row in all_variant_rows],
            "source_type",
        )
        largest_family_share = (
            max(source_family_counts.values()) / len(all_variant_rows)
            if all_variant_rows and source_family_counts
            else 0.0
        )
        normalization_applied_rows = sum(int(bool(row.get("style_normalization_applied"))) for row in all_variant_rows)
        variant_reports[variant_name] = {
            "variant_root": str(variant_root),
            "split_counts": split_counts,
            "rows_by_source_family": source_family_counts,
            "rows_by_source_type": source_type_counts,
            "largest_family_share": largest_family_share,
            "source_family_proportions": _share_by(all_variant_rows, "source_family"),
            "source_type_proportions": _share_by(
                [{**row, "source_type": _source_type(row)} for row in all_variant_rows],
                "source_type",
            ),
            "normalization_applied_rows": normalization_applied_rows,
            "cap_report": cap_report,
        }
        diff_payload["variants"][variant_name] = {
            "total_rows": len(all_variant_rows),
            "family_count": len(source_family_counts),
            "largest_family_share": largest_family_share,
            "normalization_applied_rows": normalization_applied_rows,
        }

    manifest = {
        "input_root": str(input_path),
        "primary_variant": balanced_config.primary_variant,
        "config": {
            "family_caps": balanced_config.family_caps,
            "source_type_caps": balanced_config.source_type_caps,
            "salt": balanced_config.salt,
        },
        "variants": variant_reports,
    }
    manifest_path = manifests_dir / "balanced_sidecar_manifest.yaml"
    report_json_path = reports_dir / "balanced_sidecar_report.json"
    report_md_path = reports_dir / "balanced_sidecar_report.md"
    diff_json_path = reports_dir / "v4_multisource_vs_v5_balanced_diff.json"
    write_yaml(manifest_path, manifest)
    report_json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    diff_json_path.write_text(json.dumps(diff_payload, indent=2), encoding="utf-8")

    lines = ["# Balanced Multisource Sidecar Report", ""]
    lines.append(f"- input_total_rows: {len(all_input_rows)}")
    lines.append(f"- input_largest_family_share: {input_family_share:.4f}")
    for variant_name, payload in variant_reports.items():
        lines.append("")
        lines.append(f"## {variant_name}")
        lines.append(f"- split_counts: {payload['split_counts']}")
        lines.append(f"- largest_family_share: {payload['largest_family_share']:.4f}")
        lines.append(f"- rows_by_source_type: {payload['rows_by_source_type']}")
        lines.append(f"- normalization_applied_rows: {payload['normalization_applied_rows']}")
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "manifest_path": str(manifest_path),
        "report_json": str(report_json_path),
        "report_md": str(report_md_path),
        "diff_json": str(diff_json_path),
        "variants": variant_reports,
    }
