"""Reporting helpers for week-5 standalone probe corpora."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import yaml

from modalchess.data.preprocessing_common import iter_records_from_path
from modalchess.data.probe_corpora import generate_probe_rationale_readiness


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    return [dict(row) for row in iter_records_from_path(path_obj)]


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _markdown_from_probe_report(report: dict[str, Any]) -> str:
    lines = ["# Probe Corpora Report", ""]
    lines.append("## Counts")
    for source_name, split_counts in report["counts_by_source_split"].items():
        lines.append(
            f"- `{source_name}`: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}"
        )
    lines.append("")
    lines.append("## Split Hygiene")
    for source_name, strategy in report.get("split_strategy_by_source", {}).items():
        lines.append(
            f"- `{source_name}`: split_key_type={strategy['split_key_type']}, "
            f"candidate_game_id_rows={strategy['candidate_game_id_rows']}, "
            f"repeated_group_count={strategy['repeated_group_count']}, "
            f"max_group_size={strategy['max_group_size']}"
        )
        if strategy.get("fallback_reason"):
            lines.append(f"  fallback_reason={strategy['fallback_reason']}")
    lines.append("")
    lines.append("## Empty Text Rates")
    for source_name, rate in report["empty_text_rates"].items():
        lines.append(f"- `{source_name}`: {rate:.6f}")
    lines.append("")
    lines.append("## Rare Label Warnings")
    if report["rare_label_warnings"]:
        for warning in report["rare_label_warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- No rare-label warnings.")
    return "\n".join(lines) + "\n"


def _markdown_from_rationale_report(report: dict[str, Any]) -> str:
    lines = ["# Rationale Readiness Report", ""]
    lines.append(f"- total_rows: {report['total_rows']}")
    lines.append(f"- theme_tag_rows: {report['theme_tag_rows']}")
    lines.append(f"- duplicate_probe_id_count: {report['duplicate_probe_id_count']}")
    lines.append("")
    lines.append("## Source Mix")
    for source_name, count in report["source_mix"].items():
        lines.append(f"- `{source_name}`: {count}")
    lines.append("")
    lines.append("## Move-Conditioned Coverage")
    for split_name, count in report["move_conditioned_rows"].items():
        lines.append(f"- `{split_name}`: {count}")
    lines.append("")
    lines.append("## Empty Text Rates")
    for split_name, rate in report["empty_text_rates"].items():
        lines.append(f"- `{split_name}`: {rate:.6f}")
    return "\n".join(lines) + "\n"


def compare_probe_split_roots(
    *,
    previous_root: str | Path,
    current_root: str | Path,
) -> dict[str, Any]:
    """Compare split assignments between two probe corpus roots."""
    previous_root_path = Path(previous_root)
    current_root_path = Path(current_root)
    source_names = ("mate", "puzzle")
    source_name_map = {
        "mate": "mate",
        "puzzle": "lichess_puzzle",
    }
    source_diffs: dict[str, Any] = {}
    total_rows_compared = 0
    total_rows_with_split_change = 0

    for source_name in source_names:
        previous_rows: dict[str, dict[str, Any]] = {}
        current_rows: dict[str, dict[str, Any]] = {}
        for split_name in ("train", "val", "test"):
            for row in _load_rows(previous_root_path / f"{source_name}_{split_name}.jsonl"):
                previous_rows[str(row["source_row_id"])] = row
            for row in _load_rows(current_root_path / f"{source_name}_{split_name}.jsonl"):
                current_rows[str(row["source_row_id"])] = row

        shared_ids = sorted(set(previous_rows) & set(current_rows))
        changed_ids = [
            source_row_id
            for source_row_id in shared_ids
            if str(previous_rows[source_row_id].get("split")) != str(current_rows[source_row_id].get("split"))
        ]
        total_rows_compared += len(shared_ids)
        total_rows_with_split_change += len(changed_ids)
        source_diffs[source_name_map[source_name]] = {
            "rows_compared": len(shared_ids),
            "rows_with_split_change": len(changed_ids),
            "rows_only_in_previous": len(set(previous_rows) - set(current_rows)),
            "rows_only_in_current": len(set(current_rows) - set(previous_rows)),
            "changed_source_row_ids_sample": changed_ids[:50],
        }

    previous_manifest = _load_manifest(previous_root_path / "manifests" / "probe_manifest.yaml")
    current_manifest = _load_manifest(current_root_path / "manifests" / "probe_manifest.yaml")
    note = (
        "Split assignments changed for some rows after switching to game-aware grouping."
        if total_rows_with_split_change > 0
        else "No split assignments changed; current sources fell back to source_row_id because no trustworthy repeated game_id groups were available."
    )
    return {
        "previous_root": str(previous_root_path),
        "current_root": str(current_root_path),
        "source_diffs": source_diffs,
        "total_rows_compared": total_rows_compared,
        "total_rows_with_split_change": total_rows_with_split_change,
        "previous_split_strategy_by_source": previous_manifest.get("split_strategy_by_source", {}),
        "current_split_strategy_by_source": current_manifest.get("split_strategy_by_source", {}),
        "potential_leakage_reduction_note": note,
    }


def generate_probe_corpora_report(*, input_root: str | Path) -> dict[str, Any]:
    """Generate standalone probe corpus QA report."""
    root = Path(input_root)
    manifest = _load_manifest(root / "manifests" / "probe_manifest.yaml")
    counts_by_source_split: dict[str, dict[str, int]] = {}
    empty_text_rates: dict[str, float] = {}
    label_counts: dict[str, dict[str, int]] = {}
    rare_label_warnings: list[str] = []
    null_target_move_rates: dict[str, float] = {}
    target_label_coverage: dict[str, int] = {}

    for source_name in ("mate", "puzzle"):
        source_rows = []
        for split_name in ("train", "val", "test"):
            rows = _load_rows(root / f"{source_name}_{split_name}.jsonl")
            counts_by_source_split.setdefault(source_name, {})[split_name] = len(rows)
            source_rows.extend(rows)
        empty_text_count = sum(
            int(not (row.get("strategy_text") or row.get("tactic_text")))
            for row in source_rows
        )
        empty_text_rates[source_name] = (empty_text_count / len(source_rows)) if source_rows else 1.0
        null_target_move_rates[source_name] = (
            sum(int(not row.get("target_move_uci")) for row in source_rows) / len(source_rows)
        ) if source_rows else 1.0

        source_target_rows = []
        for split_name in ("train", "val", "test"):
            source_target_rows.extend(_load_rows(root / f"{source_name}_targets_{split_name}.jsonl"))
        counter: Counter[str] = Counter()
        non_empty_target_rows = 0
        for row in source_target_rows:
            labels = [str(label) for label in row.get("target_labels", [])]
            non_empty_target_rows += int(bool(labels))
            for label in labels:
                counter[label] += 1
        label_counts[source_name] = dict(sorted(counter.items()))
        target_label_coverage[source_name] = non_empty_target_rows

    rare_label_warnings = []
    for source_name, counts in label_counts.items():
        for label_name, count in counts.items():
            if count < 20:
                rare_label_warnings.append(f"{source_name} label `{label_name}` is rare ({count}).")

    return {
        "input_root": str(root),
        "split_strategy_by_source": manifest.get("split_strategy_by_source", {}),
        "counts_by_source_split": counts_by_source_split,
        "empty_text_rates": empty_text_rates,
        "null_target_move_rates": null_target_move_rates,
        "label_counts": label_counts,
        "target_label_coverage": target_label_coverage,
        "rare_label_warnings": rare_label_warnings,
    }


def write_probe_reports(
    *,
    input_root: str | Path,
    output_dir: str | Path,
    compare_root: str | Path | None = None,
) -> dict[str, str]:
    """Write both probe corpus and rationale-readiness reports."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    probe_report = generate_probe_corpora_report(input_root=input_root)
    rationale_report = generate_probe_rationale_readiness(input_root=input_root)

    probe_json = output_root / "probe_corpora_report.json"
    probe_md = output_root / "probe_corpora_report.md"
    rationale_json = output_root / "rationale_readiness_report.json"
    rationale_md = output_root / "rationale_readiness_report.md"
    diff_json = output_root / "v1_vs_v2_split_diff.json"

    probe_json.write_text(json.dumps(probe_report, indent=2), encoding="utf-8")
    probe_md.write_text(_markdown_from_probe_report(probe_report), encoding="utf-8")
    rationale_json.write_text(json.dumps(rationale_report, indent=2), encoding="utf-8")
    rationale_md.write_text(_markdown_from_rationale_report(rationale_report), encoding="utf-8")
    if compare_root is not None:
        diff_payload = compare_probe_split_roots(previous_root=compare_root, current_root=input_root)
        diff_json.write_text(json.dumps(diff_payload, indent=2), encoding="utf-8")
    return {
        "probe_json": str(probe_json),
        "probe_md": str(probe_md),
        "rationale_json": str(rationale_json),
        "rationale_md": str(rationale_md),
        "diff_json": str(diff_json) if compare_root is not None else "",
    }
