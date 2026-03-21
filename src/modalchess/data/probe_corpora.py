"""Standalone language probe corpora builders for week-5."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from modalchess.data.preprocessing_common import (
    StableSplitConfig,
    assign_split_by_game_id,
    iter_records_from_path,
    normalize_fen_for_eval_join,
    parse_space_or_comma_separated,
    stable_hash_text,
    write_jsonl,
    write_yaml,
)


@dataclass(slots=True)
class ProbeCorpusConfig:
    """Standalone probe corpus build configuration."""

    split_config: StableSplitConfig = field(
        default_factory=lambda: StableSplitConfig(salt="modalchess_week5_probe")
    )


def _source_row_id(row: Mapping[str, Any]) -> str:
    for key in ("source_row_id", "position_id", "id", "game_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    raise ValueError("source_row_id를 추출할 수 없다.")


def _normalize_mate_row(row: Mapping[str, Any], config: ProbeCorpusConfig) -> dict[str, Any]:
    source_row_id = _source_row_id(row)
    split_name = assign_split_by_game_id(source_row_id, config.split_config)
    fen = str(row["fen"])
    return {
        "probe_id": stable_hash_text(f"mate:{source_row_id}", prefix="probe_", length=16),
        "source": "mate",
        "source_row_id": source_row_id,
        "split": split_name,
        "fen": fen,
        "fen_4field": normalize_fen_for_eval_join(fen),
        "target_move_uci": None,
        "candidate_moves": parse_space_or_comma_separated(row.get("candidate_moves")),
        "strategy_text": row.get("strategy_text"),
        "tactic_text": row.get("tactic_text"),
        "theme_tags": None,
        "history_fens": row.get("history_fens") or [fen],
        "metadata": {
            "game_id": row.get("game_id"),
            "preferred_move": row.get("preferred_move"),
            "original_source": row.get("source", "mate"),
        },
    }


def _normalize_puzzle_row(row: Mapping[str, Any], config: ProbeCorpusConfig) -> dict[str, Any]:
    source_row_id = _source_row_id(row)
    split_name = assign_split_by_game_id(source_row_id, config.split_config)
    fen = str(row["fen"])
    return {
        "probe_id": stable_hash_text(f"puzzle:{source_row_id}", prefix="probe_", length=16),
        "source": "lichess_puzzle",
        "source_row_id": source_row_id,
        "split": split_name,
        "fen": fen,
        "fen_4field": normalize_fen_for_eval_join(fen),
        "target_move_uci": row.get("target_move_uci"),
        "candidate_moves": None,
        "strategy_text": None,
        "tactic_text": None,
        "theme_tags": parse_space_or_comma_separated(row.get("theme_tags") or row.get("concept_tags")),
        "history_fens": row.get("history_fens") or [fen],
        "metadata": {
            "game_id": row.get("game_id"),
            "next_fen": row.get("next_fen"),
            "original_source": row.get("source", "lichess_puzzle"),
        },
    }


def build_probe_corpora(
    *,
    mate_path: str | Path,
    puzzle_path: str | Path,
    output_root: str | Path,
    config: ProbeCorpusConfig | None = None,
) -> dict[str, Any]:
    """Build deterministic standalone MATE and puzzle probe corpora."""
    corpus_config = config or ProbeCorpusConfig()
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    normalized_rows = {
        "mate": [_normalize_mate_row(row, corpus_config) for row in iter_records_from_path(mate_path)],
        "puzzle": [_normalize_puzzle_row(row, corpus_config) for row in iter_records_from_path(puzzle_path)],
    }
    split_counts: dict[str, dict[str, int]] = {}
    outputs: dict[str, str] = {}

    for source_name, rows in normalized_rows.items():
        split_rows = {"train": [], "val": [], "test": []}
        for row in rows:
            split_rows[str(row["split"])].append(row)
        split_counts[source_name] = {split_name: len(split_rows[split_name]) for split_name in split_rows}
        for split_name, payload_rows in split_rows.items():
            output_path = output_dir / f"{source_name}_{split_name}.jsonl"
            write_jsonl(output_path, payload_rows)
            outputs[f"{source_name}_{split_name}"] = str(output_path)

    manifest = {
        "inputs": {
            "mate": str(mate_path),
            "puzzle": str(puzzle_path),
        },
        "config": {
            "split_config": asdict(corpus_config.split_config),
        },
        "counts": split_counts,
        "outputs": outputs,
    }
    manifest_path = manifest_dir / "probe_manifest.yaml"
    write_yaml(manifest_path, manifest)
    return {
        "manifest_path": str(manifest_path),
        "counts": split_counts,
        "outputs": outputs,
    }


def generate_probe_rationale_readiness(
    *,
    input_root: str | Path,
) -> dict[str, Any]:
    """Summarize standalone probe corpora as rationale-ready evaluation candidates."""
    root = Path(input_root)
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    move_conditioned_counts: Counter[str] = Counter()
    empty_text_counts: Counter[str] = Counter()
    theme_rows = 0
    duplicate_probe_ids: list[str] = []
    seen_probe_ids: set[str] = set()
    special_rule_counts = {"promotion": 0, "castling": 0, "en_passant": 0, "check_evasion": 0}

    from modalchess.data.preprocessing_common import special_rule_flags

    for path in sorted(root.glob("*_*.jsonl")):
        if path.name.startswith(("mate_targets_", "puzzle_targets_")):
            continue
        for row in iter_records_from_path(path):
            split_name = str(row["split"])
            source_name = str(row["source"])
            probe_id = str(row["probe_id"])
            if probe_id in seen_probe_ids:
                duplicate_probe_ids.append(probe_id)
            seen_probe_ids.add(probe_id)
            source_counts[source_name] += 1
            split_counts[split_name] += 1
            move_conditioned_counts[split_name] += int(bool(row.get("target_move_uci")))
            empty_text_counts[split_name] += int(not (row.get("strategy_text") or row.get("tactic_text")))
            theme_rows += int(bool(row.get("theme_tags")))
            flags = special_rule_flags(
                str(row["fen"]),
                str(row.get("target_move_uci")) if row.get("target_move_uci") else None,
            )
            for key, active in flags.items():
                special_rule_counts[key] += int(active)

    total_rows = sum(split_counts.values())
    return {
        "total_rows": total_rows,
        "source_mix": dict(source_counts),
        "split_counts": dict(split_counts),
        "move_conditioned_rows": dict(move_conditioned_counts),
        "empty_text_rates": {
            split_name: (empty_text_counts[split_name] / split_counts[split_name]) if split_counts[split_name] else 1.0
            for split_name in ("train", "val", "test")
        },
        "theme_tag_rows": theme_rows,
        "special_rule_coverage": special_rule_counts,
        "duplicate_probe_id_count": len(duplicate_probe_ids),
    }
