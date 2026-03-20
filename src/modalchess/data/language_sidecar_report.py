"""Week-4 language-sidecar QA reporting."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from modalchess.data.preprocessing_common import (
    iter_records_from_path,
    special_rule_flags,
)


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    return [dict(row) for row in iter_records_from_path(path_obj)]


def _text_length_stats(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    lengths: list[int] = []
    for row in rows:
        texts = [
            str(row.get("strategy_text") or ""),
            str(row.get("tactic_text") or ""),
            str(row.get("rationale_short") or ""),
        ]
        combined = " ".join(text for text in texts if text).strip()
        if combined:
            lengths.append(len(combined.split()))
    if not lengths:
        return {"count": 0, "mean_words": 0.0, "max_words": 0.0}
    return {
        "count": len(lengths),
        "mean_words": float(fmean(lengths)),
        "max_words": float(max(lengths)),
    }


def _theme_coverage(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for tag in row.get("theme_tags") or row.get("motif_tags") or []:
            counts[str(tag)] += 1
    return dict(sorted(counts.items()))


def _special_rule_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {"promotion": 0, "castling": 0, "en_passant": 0, "check_evasion": 0}
    for row in rows:
        flags = special_rule_flags(
            str(row["fen"]),
            str(row.get("target_move_uci")) if row.get("target_move_uci") else None,
        )
        for key, active in flags.items():
            counts[key] += int(active)
    return counts


def _matched_rows_by_split(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    matched: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for split_name, rows in split_rows.items():
        matched[split_name].extend([row for row in rows if row.get("matched_supervised")])
    return matched


def _leakage_summary(matched_rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    positions = {
        split_name: {str(row["matched_position_id"]) for row in rows if row.get("matched_position_id")}
        for split_name, rows in matched_rows_by_split.items()
    }
    leakage = {
        "train_val": sorted(positions["train"] & positions["val"]),
        "train_test": sorted(positions["train"] & positions["test"]),
        "val_test": sorted(positions["val"] & positions["test"]),
    }
    leakage_count = sum(len(values) for values in leakage.values())
    return {
        "split_leakage": leakage,
        "passes": leakage_count == 0,
    }


def _readiness_status(
    *,
    leakage_ok: bool,
    matched_counts: dict[str, int],
    rationale_counts: dict[str, int],
) -> tuple[str, list[str], list[str]]:
    reasons: list[str] = []
    recommendations: list[str] = []
    if not leakage_ok:
        reasons.append("split leakage detected in matched supervised positions")
    for split_name in ("train", "val", "test"):
        if matched_counts[split_name] == 0:
            reasons.append(f"matched {split_name} split is empty")
    for split_name in ("val", "test"):
        if rationale_counts[split_name] == 0:
            reasons.append(f"rationale_{split_name} is empty")

    if reasons:
        recommendations.append("Acquire more overlapping language-sidecar sources before week-5.")
        recommendations.append("Keep week-5 language work constrained to evaluation-only probes.")
        return "not_ready", reasons, recommendations

    if matched_counts["train"] < 100 or matched_counts["val"] < 25 or matched_counts["test"] < 25:
        reasons.append("matched sidecar coverage is non-zero but too small for stable week-5 alignment training")
        recommendations.append("Acquire more language-sidecar data or broader safe matching sources.")
        recommendations.append("If week-5 proceeds, keep it to constrained evaluation-only or retrieval-style probes.")
        return "partially_ready", reasons, recommendations

    recommendations.append("Frozen-backbone language alignment experiments are unblocked for week-5.")
    return "ready", reasons, recommendations


def _markdown_from_report(report: dict[str, Any]) -> str:
    lines = ["# Language Sidecar Report", ""]
    readiness = report["readiness"]
    lines.append(f"- readiness: `{readiness['status']}`")
    lines.append(f"- leakage_check: `{'PASS' if report['leakage_check']['passes'] else 'FAIL'}`")
    lines.append("")
    lines.append("## Matched Counts")
    for split_name in ("train", "val", "test"):
        lines.append(f"- `{split_name}`: {report['matched_counts_by_split'][split_name]}")
    lines.append("")
    lines.append("## Match Rates")
    lines.append(f"- exact_full_fen_match_rate: {report['match_rates']['full_fen_match_rate']:.6f}")
    lines.append(f"- fen_4field_match_rate: {report['match_rates']['fen_4field_match_rate']:.6f}")
    lines.append(f"- move_conditioned_match_rate: {report['match_rates']['move_conditioned_match_rate']:.6f}")
    lines.append("")
    lines.append("## Special-Rule Coverage")
    for split_name in ("train", "val", "test"):
        subset_counts = report["special_rule_coverage"].get(split_name, {})
        lines.append(
            f"- `{split_name}`: promotion={subset_counts.get('promotion', 0)}, "
            f"castling={subset_counts.get('castling', 0)}, "
            f"en_passant={subset_counts.get('en_passant', 0)}, "
            f"check_evasion={subset_counts.get('check_evasion', 0)}"
        )
    lines.append("")
    lines.append("## Reasons")
    if readiness["reasons"]:
        for reason in readiness["reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- No blocking reasons.")
    lines.append("")
    lines.append("## Recommendations")
    for recommendation in readiness["recommendations"]:
        lines.append(f"- {recommendation}")
    return "\n".join(lines) + "\n"


def generate_language_sidecar_report(
    *,
    input_root: str | Path,
) -> dict[str, Any]:
    """Generate week-4 language-sidecar QA report."""
    root = Path(input_root)
    split_rows = {
        "train": _load_rows(root / "mate_matched_train.jsonl") + _load_rows(root / "puzzle_matched_train.jsonl"),
        "val": _load_rows(root / "mate_matched_val.jsonl") + _load_rows(root / "puzzle_matched_val.jsonl"),
        "test": _load_rows(root / "mate_matched_test.jsonl") + _load_rows(root / "puzzle_matched_test.jsonl"),
    }
    unmatched_rows = _load_rows(root / "mate_unmatched.jsonl")
    rationale_rows = {
        "train": _load_rows(root / "rationale_train.jsonl"),
        "val": _load_rows(root / "rationale_val.jsonl"),
        "test": _load_rows(root / "rationale_test.jsonl"),
    }
    matched_rows_by_split = _matched_rows_by_split(split_rows)
    matched_counts = {
        split_name: len(rows)
        for split_name, rows in matched_rows_by_split.items()
    }
    total_sidecar_rows = sum(len(rows) for rows in split_rows.values()) + len(unmatched_rows)
    exact_matches = sum(
        1
        for rows in split_rows.values()
        for row in rows
        if row.get("alignment_type") == "fen_exact"
    )
    fen4_matches = sum(
        1
        for rows in split_rows.values()
        for row in rows
        if row.get("alignment_type") == "fen_4field"
    )
    move_conditioned_matches = sum(
        1
        for rows in split_rows.values()
        for row in rows
        if str(row.get("alignment_type", "")).endswith("target_move")
    )
    ambiguous_match_count = sum(
        1
        for row in unmatched_rows
        if any(
            token in str(row.get("notes", ""))
            for token in ("ambiguous", "ambiguity", "cross_split")
        )
    )
    leakage_check = _leakage_summary(matched_rows_by_split)
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for split_name, rows in split_rows.items():
        split_counts[split_name] += len(rows)
        for row in rows:
            source_counts[str(row["source"])] += 1
    for row in unmatched_rows:
        source_counts[str(row["source"])] += 1

    empty_text_count = 0
    null_field_rates: dict[str, float] = {}
    text_rows = []
    for split_name in ("train", "val", "test"):
        text_rows.extend(split_rows[split_name])
        text_rows.extend(rationale_rows[split_name])
    for row in text_rows:
        has_text = any(
            bool(row.get(field_name))
            for field_name in ("strategy_text", "tactic_text", "rationale_short")
        )
        empty_text_count += int(not has_text)
    if text_rows:
        null_field_rates = {
            "empty_text_rate": empty_text_count / len(text_rows),
            "null_theme_tags_rate": sum(int(not row.get("theme_tags")) for row in text_rows) / len(text_rows),
            "null_target_move_rate": sum(int(not row.get("target_move_uci")) for row in text_rows) / len(text_rows),
        }
    else:
        null_field_rates = {
            "empty_text_rate": 1.0,
            "null_theme_tags_rate": 1.0,
            "null_target_move_rate": 1.0,
        }

    rationale_counts = {split_name: len(rows) for split_name, rows in rationale_rows.items()}
    readiness_status, readiness_reasons, recommendations = _readiness_status(
        leakage_ok=leakage_check["passes"],
        matched_counts=matched_counts,
        rationale_counts=rationale_counts,
    )

    report = {
        "input_root": str(root),
        "counts_by_source": dict(source_counts),
        "counts_by_split": dict(split_counts),
        "matched_counts_by_split": matched_counts,
        "matched_vs_unmatched": {
            "matched_rows": sum(matched_counts.values()),
            "unmatched_rows": len(unmatched_rows),
        },
        "match_rates": {
            "full_fen_match_rate": (exact_matches / total_sidecar_rows) if total_sidecar_rows else 0.0,
            "fen_4field_match_rate": (fen4_matches / total_sidecar_rows) if total_sidecar_rows else 0.0,
            "move_conditioned_match_rate": (
                move_conditioned_matches / total_sidecar_rows
            ) if total_sidecar_rows else 0.0,
        },
        "duplicate_or_ambiguous_match_count": ambiguous_match_count,
        "text_length_stats": _text_length_stats(text_rows),
        "null_field_rates": null_field_rates,
        "theme_tag_coverage": _theme_coverage(text_rows),
        "special_rule_coverage": {
            split_name: _special_rule_counts(rows)
            for split_name, rows in matched_rows_by_split.items()
        },
        "leakage_check": leakage_check,
        "rationale_counts": rationale_counts,
        "readiness": {
            "status": readiness_status,
            "reasons": readiness_reasons,
            "recommendations": recommendations,
        },
    }
    return report


def write_language_sidecar_report(
    *,
    input_root: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON and Markdown QA reports for week-4 sidecars."""
    report = generate_language_sidecar_report(input_root=input_root)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "language_sidecar_report.json"
    md_path = output_root / "language_sidecar_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_from_report(report), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}
