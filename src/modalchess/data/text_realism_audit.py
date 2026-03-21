"""Audit how realistic current language probes are with respect to raw text."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

import yaml

from modalchess.data.preprocessing_common import iter_records_from_path


def _load_yaml(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        return {}
    with input_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _load_json(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        return {}
    return json.loads(input_path.read_text(encoding="utf-8"))


def _puzzle_retrieval_vocab_size(rows: list[dict[str, Any]], min_support: int) -> int:
    counter: Counter[str] = Counter()
    for row in rows:
        for label in row.get("target_labels", []):
            counter[str(label)] += 1
        for field_name, token_name in (
            ("promotion_flag", "promotion_flag"),
            ("castling_flag", "castling_flag"),
            ("en_passant_flag", "en_passant_flag"),
            ("check_evasion_flag", "check_evasion_flag"),
        ):
            if row.get(field_name):
                counter[token_name] += 1
    return sum(1 for count in counter.values() if count >= min_support)


def _mate_retrieval_vocab_size(rows: list[dict[str, Any]], min_support: int) -> int:
    counter: Counter[str] = Counter()
    for row in rows:
        for label in row.get("target_labels", []):
            counter[str(label)] += 1
    return sum(1 for count in counter.values() if count >= min_support)


def _source_rows(input_root: Path, source_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        path = input_root / f"{source_name}_{split_name}.jsonl"
        if not path.exists():
            continue
        rows.extend(dict(row) for row in iter_records_from_path(path))
    return rows


def _target_rows(input_root: Path, source_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        path = input_root / f"{source_name}_targets_{split_name}.jsonl"
        if not path.exists():
            continue
        rows.extend(dict(row) for row in iter_records_from_path(path))
    return rows


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Text Realism Audit", ""]
    for source_name in ("mate", "puzzle"):
        entry = report["sources"][source_name]
        lines.append(f"## {source_name}")
        lines.append(f"- total_rows: {entry['total_rows']}")
        lines.append(f"- raw_text_rows: {entry['raw_text_rows']}")
        lines.append(f"- natural_text_available: {entry['natural_text_available']}")
        lines.append(f"- classification_label_vocab_size: {entry['classification_label_vocab_size']}")
        lines.append(f"- retrieval_token_vocab_size: {entry['retrieval_token_vocab_size']}")
        lines.append(f"- current_classification_representation: {entry['current_classification_representation']}")
        lines.append(f"- current_retrieval_representation: {entry['current_retrieval_representation']}")
        lines.append(f"- uses_raw_text_in_week6_classification: {entry['uses_raw_text_in_week6_classification']}")
        lines.append(f"- uses_raw_text_in_week6_retrieval: {entry['uses_raw_text_in_week6_retrieval']}")
        lines.append("")
    lines.append("## Conclusions")
    for conclusion in report["conclusions"]:
        lines.append(f"- {conclusion}")
    return "\n".join(lines) + "\n"


def audit_probe_text_realism(
    *,
    input_root: str | Path = "data/pilot/language_probe_v2",
    week6_readiness_path: str | Path = "outputs/week6/readiness_probes/probe_results.json",
    week6_retrieval_path: str | Path = "outputs/week6/retrieval_probes/retrieval_results.json",
    output_dir: str | Path | None = None,
    retrieval_min_support: int = 25,
) -> dict[str, Any]:
    """Audit whether current probes use real text or derived token space."""
    root = Path(input_root)
    report_dir = Path(output_dir) if output_dir is not None else root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    target_manifest = _load_yaml(root / "manifests" / "probe_targets_manifest.yaml")
    readiness_payload = _load_json(week6_readiness_path)
    retrieval_payload = _load_json(week6_retrieval_path)

    mate_rows = _source_rows(root, "mate")
    puzzle_rows = _source_rows(root, "puzzle")
    mate_target_rows = _target_rows(root, "mate")
    puzzle_target_rows = _target_rows(root, "puzzle")

    label_freqs = target_manifest.get("label_frequencies_by_source", {}) or target_manifest.get("label_counts_by_source", {})
    mate_label_vocab_size = len(label_freqs.get("mate", {}))
    puzzle_label_vocab_size = len(label_freqs.get("puzzle", {}))

    mate_entry = {
        "total_rows": len(mate_rows),
        "raw_text_rows": sum(
            int(bool(row.get("strategy_text") or row.get("tactic_text")))
            for row in mate_rows
        ),
        "natural_text_available": True,
        "classification_label_vocab_size": mate_label_vocab_size,
        "retrieval_token_vocab_size": _mate_retrieval_vocab_size(mate_target_rows, retrieval_min_support),
        "current_classification_representation": "conservative keyword-derived multi-label targets",
        "current_retrieval_representation": "derived keyword-token bag retrieved from frozen board probes",
        "uses_raw_text_in_week6_classification": False,
        "uses_raw_text_in_week6_retrieval": False,
        "week6_best_classification_backbone": next(
            (
                row["backbone"]
                for row in readiness_payload.get("aggregate", [])
                if row.get("family") == "mate"
                and row.get("probe_model") == "mlp"
                and row.get("test_micro_average_precision_mean")
                == max(
                    candidate.get("test_micro_average_precision_mean", 0.0)
                    for candidate in readiness_payload.get("aggregate", [])
                    if candidate.get("family") == "mate" and candidate.get("probe_model") != "baseline"
                )
            ),
            None,
        ),
        "week6_best_retrieval_backbone": next(
            (
                row["backbone"]
                for row in retrieval_payload.get("aggregate", [])
                if row.get("family") == "mate"
                and row.get("board_to_text_mrr_mean")
                == max(
                    candidate.get("board_to_text_mrr_mean", 0.0)
                    for candidate in retrieval_payload.get("aggregate", [])
                    if candidate.get("family") == "mate"
                )
            ),
            None,
        ),
    }
    puzzle_entry = {
        "total_rows": len(puzzle_rows),
        "raw_text_rows": 0,
        "natural_text_available": False,
        "classification_label_vocab_size": puzzle_label_vocab_size,
        "retrieval_token_vocab_size": _puzzle_retrieval_vocab_size(puzzle_target_rows, retrieval_min_support),
        "current_classification_representation": "theme-tag multi-label targets",
        "current_retrieval_representation": "synthetic theme/special-rule token string retrieval",
        "uses_raw_text_in_week6_classification": False,
        "uses_raw_text_in_week6_retrieval": False,
        "week6_best_classification_backbone": next(
            (
                row["backbone"]
                for row in readiness_payload.get("aggregate", [])
                if row.get("family") == "puzzle"
                and row.get("probe_model") == "mlp"
                and row.get("test_micro_average_precision_mean")
                == max(
                    candidate.get("test_micro_average_precision_mean", 0.0)
                    for candidate in readiness_payload.get("aggregate", [])
                    if candidate.get("family") == "puzzle" and candidate.get("probe_model") != "baseline"
                )
            ),
            None,
        ),
        "week6_best_retrieval_backbone": next(
            (
                row["backbone"]
                for row in retrieval_payload.get("aggregate", [])
                if row.get("family") == "puzzle"
                and row.get("board_to_text_mrr_mean")
                == max(
                    candidate.get("board_to_text_mrr_mean", 0.0)
                    for candidate in retrieval_payload.get("aggregate", [])
                    if candidate.get("family") == "puzzle"
                )
            ),
            None,
        ),
    }

    report = {
        "input_root": str(root),
        "sources": {
            "mate": mate_entry,
            "puzzle": puzzle_entry,
        },
        "conclusions": [
            "MATE rows contain real strategy/tactic text, but week-6 classification and retrieval mostly operated in conservative keyword-derived label/token space.",
            "Puzzle rows do not contain natural free text; week-6 probes used theme tags and synthetic special-rule tag strings rather than natural language.",
            "Week-6 probe gains should be interpreted as language-adjacent representation evidence, not as evidence of natural-language understanding.",
        ],
    }

    json_path = report_dir / "text_realism_audit.json"
    md_path = report_dir / "text_realism_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "report": report,
    }
