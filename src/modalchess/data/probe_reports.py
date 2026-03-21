"""Reporting helpers for week-5 standalone probe corpora."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from modalchess.data.preprocessing_common import iter_records_from_path, write_yaml
from modalchess.data.probe_corpora import generate_probe_rationale_readiness


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    return [dict(row) for row in iter_records_from_path(path_obj)]


def _markdown_from_probe_report(report: dict[str, Any]) -> str:
    lines = ["# Probe Corpora Report", ""]
    lines.append("## Counts")
    for source_name, split_counts in report["counts_by_source_split"].items():
        lines.append(
            f"- `{source_name}`: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}"
        )
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


def generate_probe_corpora_report(*, input_root: str | Path) -> dict[str, Any]:
    """Generate standalone probe corpus QA report."""
    root = Path(input_root)
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

    probe_json.write_text(json.dumps(probe_report, indent=2), encoding="utf-8")
    probe_md.write_text(_markdown_from_probe_report(probe_report), encoding="utf-8")
    rationale_json.write_text(json.dumps(rationale_report, indent=2), encoding="utf-8")
    rationale_md.write_text(_markdown_from_rationale_report(rationale_report), encoding="utf-8")
    return {
        "probe_json": str(probe_json),
        "probe_md": str(probe_md),
        "rationale_json": str(rationale_json),
        "rationale_md": str(rationale_md),
    }
