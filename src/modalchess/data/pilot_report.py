"""real pilot 데이터셋 QA 리포트 생성기."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import chess

from modalchess.data.preprocessing_common import (
    iter_records_from_path,
    special_rule_flags,
    validate_modalchess_record,
)
from modalchess.utils.config import load_yaml_config


def _classify_validation_error(row: dict[str, Any], exc: Exception) -> str:
    try:
        chess.Board(str(row.get("fen", "")))
    except Exception:
        return "invalid_fen"
    message = str(exc)
    if "target_move_uci" in message:
        return "invalid_target_move_uci"
    if "history_fens" in message:
        return "history_fens_contract_violation"
    return "validation_error"


def _default_source_name(split_name: str) -> str:
    if split_name in {"train", "val", "test"}:
        return "lichess_pgn"
    return "unknown"


def _load_drop_reasons(manifest_dir: Path) -> dict[str, int]:
    totals: Counter[str] = Counter()
    if not manifest_dir.exists():
        return {}
    for manifest_path in sorted(manifest_dir.glob("*.yaml")):
        manifest = load_yaml_config(manifest_path)
        report = manifest.get("report", {})
        for reason, count in report.get("drop_reasons", {}).items():
            totals[str(reason)] += int(count)
    return dict(totals)


def _markdown_from_report(report: dict[str, Any]) -> str:
    lines = ["# Pilot Data Report", ""]
    lines.append("## Gate")
    for key, passed in report["gate_checks"].items():
        lines.append(f"- `{key}`: {'PASS' if passed else 'FAIL'}")
    lines.append("")
    lines.append("## Split Counts")
    for split_name, split_report in report["splits"].items():
        lines.append(
            f"- `{split_name}`: rows={split_report['row_count']}, unique_games={split_report['unique_game_ids']}, "
            f"avg_history_length={split_report['average_history_length']:.2f}"
        )
    lines.append("")
    lines.append("## Subset Coverage")
    for split_name, split_report in report["splits"].items():
        subset_counts = split_report["subset_counts"]
        lines.append(
            f"- `{split_name}`: promotion={subset_counts['promotion']}, castling={subset_counts['castling']}, "
            f"en_passant={subset_counts['en_passant']}, check_evasion={subset_counts['check_evasion']}"
        )
    lines.append("")
    lines.append("## Missing Labels")
    for split_name, split_report in report["splits"].items():
        lines.append(
            f"- `{split_name}`: engine_eval_cp_missing_rate={split_report['engine_eval_cp_missing_rate']:.4f}, "
            f"concept_tags_missing_rate={split_report['concept_tags_missing_rate']:.4f}"
        )
    lines.append("")
    lines.append("## Dropped Rows")
    if report["dropped_row_counts_by_reason"]:
        for reason, count in report["dropped_row_counts_by_reason"].items():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- No drop reasons found in manifests.")
    lines.append("")
    lines.append("## Warnings")
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- No warnings.")
    return "\n".join(lines) + "\n"


def generate_pilot_data_report(
    *,
    input_root: str | Path,
    min_val_rows: int = 1000,
    min_test_rows: int = 1000,
) -> dict[str, Any]:
    """train/val/test supervised JSONL에 대한 QA 리포트를 만든다."""
    root = Path(input_root)
    manifests_dir = root / "manifests"
    split_paths = {
        "train": root / "supervised_train.jsonl",
        "val": root / "supervised_val.jsonl",
        "test": root / "supervised_test.jsonl",
    }

    report: dict[str, Any] = {
        "input_root": str(root),
        "splits": {},
        "dropped_row_counts_by_reason": _load_drop_reasons(manifests_dir),
        "warnings": [],
    }
    game_ids_by_split: dict[str, set[str]] = {}
    invalid_fen_rows = 0
    invalid_target_rows = 0
    history_contract_violations = 0

    for split_name, split_path in split_paths.items():
        row_count = 0
        unique_game_ids: set[str] = set()
        subset_counts = {"promotion": 0, "castling": 0, "en_passant": 0, "check_evasion": 0}
        source_counts: Counter[str] = Counter()
        invalid_reasons: Counter[str] = Counter()
        engine_eval_missing = 0
        concept_tags_missing = 0
        history_length_total = 0

        if not split_path.exists():
            report["warnings"].append(f"Missing split file: {split_path.name}")
            report["splits"][split_name] = {
                "row_count": 0,
                "unique_game_ids": 0,
                "subset_counts": subset_counts,
                "source_counts": {},
                "engine_eval_cp_missing_rate": 1.0,
                "concept_tags_missing_rate": 1.0,
                "average_history_length": 0.0,
                "validation_summary": {
                    "valid_rows": 0,
                    "invalid_rows": 0,
                    "invalid_reasons": {},
                },
            }
            game_ids_by_split[split_name] = set()
            continue

        for row in iter_records_from_path(split_path):
            row_count += 1
            game_id = str(row.get("game_id"))
            unique_game_ids.add(game_id)
            source_counts[str(row.get("source") or _default_source_name(split_name))] += 1
            history_fens = row.get("history_fens") or [row["fen"]]
            history_length_total += len(history_fens)
            engine_eval_missing += int(row.get("engine_eval_cp") is None)
            concept_tags_missing += int(row.get("concept_tags") is None)
            flags = special_rule_flags(str(row["fen"]), str(row.get("target_move_uci")))
            for key, active in flags.items():
                subset_counts[key] += int(active)
            try:
                validate_modalchess_record(row, require_target_move=True)
            except Exception as exc:
                reason = _classify_validation_error(row, exc)
                invalid_reasons[reason] += 1
                invalid_fen_rows += int(reason == "invalid_fen")
                invalid_target_rows += int(reason == "invalid_target_move_uci")
                history_contract_violations += int(reason == "history_fens_contract_violation")

        split_report = {
            "row_count": row_count,
            "unique_game_ids": len(unique_game_ids),
            "subset_counts": subset_counts,
            "source_counts": dict(source_counts),
            "engine_eval_cp_missing_rate": (engine_eval_missing / row_count) if row_count else 1.0,
            "concept_tags_missing_rate": (concept_tags_missing / row_count) if row_count else 1.0,
            "average_history_length": (history_length_total / row_count) if row_count else 0.0,
            "validation_summary": {
                "valid_rows": row_count - sum(invalid_reasons.values()),
                "invalid_rows": sum(invalid_reasons.values()),
                "invalid_reasons": dict(invalid_reasons),
            },
        }
        report["splits"][split_name] = split_report
        game_ids_by_split[split_name] = unique_game_ids

    split_leakage = {
        "train_val": sorted(game_ids_by_split["train"] & game_ids_by_split["val"]),
        "train_test": sorted(game_ids_by_split["train"] & game_ids_by_split["test"]),
        "val_test": sorted(game_ids_by_split["val"] & game_ids_by_split["test"]),
    }
    total_leakage = sum(len(values) for values in split_leakage.values())

    val_subset_total = sum(report["splits"]["val"]["subset_counts"].values())
    test_subset_total = sum(report["splits"]["test"]["subset_counts"].values())
    if val_subset_total < 50 or test_subset_total < 50:
        report["warnings"].append(
            "Special-rule coverage is weak in val/test; consider using puzzle_eval as an auxiliary evaluation set."
        )
    for split_name in ("val", "test"):
        for subset_name, count in report["splits"][split_name]["subset_counts"].items():
            if count < 10:
                report["warnings"].append(
                    f"{split_name} subset `{subset_name}` count is low ({count}); interpret special-rule comparisons carefully."
                )
        if report["splits"][split_name]["engine_eval_cp_missing_rate"] > 0.95:
            report["warnings"].append(
                f"{split_name} engine_eval_cp coverage is extremely sparse; prefer non-enriched training configs for week1."
            )
        if report["splits"][split_name]["concept_tags_missing_rate"] > 0.95:
            report["warnings"].append(
                f"{split_name} concept_tags coverage is extremely sparse in the backbone PGN pilot."
            )

    gate_checks = {
        "no_invalid_fen_rows": invalid_fen_rows == 0,
        "no_invalid_target_move_rows": invalid_target_rows == 0,
        "no_history_fens_contract_violations": history_contract_violations == 0,
        "no_split_leakage_by_game_id": total_leakage == 0,
        "train_non_empty": report["splits"]["train"]["row_count"] > 0,
        "val_non_empty": report["splits"]["val"]["row_count"] > 0,
        "test_non_empty": report["splits"]["test"]["row_count"] > 0,
        "val_not_tiny": report["splits"]["val"]["row_count"] >= min_val_rows,
        "test_not_tiny": report["splits"]["test"]["row_count"] >= min_test_rows,
    }
    report["validation_summary"] = {
        "invalid_fen_rows": invalid_fen_rows,
        "invalid_target_move_rows": invalid_target_rows,
        "history_fens_contract_violations": history_contract_violations,
        "split_leakage": split_leakage,
    }
    report["gate_checks"] = gate_checks
    report["passes_gate"] = all(gate_checks.values())
    return report


def write_pilot_data_report(
    *,
    input_root: str | Path,
    output_dir: str | Path,
    min_val_rows: int = 1000,
    min_test_rows: int = 1000,
) -> dict[str, str]:
    """QA 리포트를 JSON과 Markdown으로 기록한다."""
    report = generate_pilot_data_report(
        input_root=input_root,
        min_val_rows=min_val_rows,
        min_test_rows=min_test_rows,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "pilot_data_report.json"
    md_path = output_root / "pilot_data_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_from_report(report), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}
