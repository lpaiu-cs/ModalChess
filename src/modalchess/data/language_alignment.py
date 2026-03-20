"""Week-4 language sidecar alignment helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from modalchess.data.preprocessing_common import (
    iter_records_from_path,
    normalize_fen_for_eval_join,
    parse_space_or_comma_separated,
    stable_hash_text,
    write_jsonl,
    write_yaml,
)


MATCHED_ROW_FIELDS = (
    "sidecar_id",
    "source",
    "source_row_id",
    "split",
    "matched_supervised",
    "matched_position_id",
    "matched_game_id",
    "fen",
    "fen_4field",
    "target_move_uci",
    "candidate_moves",
    "strategy_text",
    "tactic_text",
    "theme_tags",
    "alignment_type",
    "alignment_confidence",
    "notes",
)


@dataclass(slots=True)
class LanguageAlignmentConfig:
    """Alignment build configuration."""

    alignment_version: str = "week4_language_v1"
    allow_fen_4field: bool = True
    allow_move_conditioned_for_puzzles: bool = True
    exact_confidence: float = 1.0
    exact_move_confidence: float = 1.0
    fen4_confidence: float = 0.75
    fen4_move_confidence: float = 0.85
    copy_optional_chessgpt: bool = True


@dataclass(slots=True)
class SupervisedSplitIndex:
    """Per-split supervised lookup tables."""

    split: str
    rows: list[dict[str, Any]]
    by_exact_fen: dict[str, list[dict[str, Any]]]
    by_exact_fen_move: dict[tuple[str, str], list[dict[str, Any]]]
    by_fen4: dict[str, list[dict[str, Any]]]
    by_fen4_move: dict[tuple[str, str], list[dict[str, Any]]]


@dataclass(slots=True)
class ResolvedAlignment:
    """Single resolved match decision."""

    split: str
    matched_record: dict[str, Any]
    alignment_type: str
    alignment_confidence: float
    notes: str | None = None


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_records_from_path(path)]


def build_supervised_indices(
    supervised_paths: Mapping[str, str | Path],
) -> dict[str, SupervisedSplitIndex]:
    """Build split-aware exact/FEN4 lookup tables for supervised JSONL."""
    indices: dict[str, SupervisedSplitIndex] = {}
    for split_name, path in supervised_paths.items():
        rows = _load_records(path)
        by_exact_fen: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_exact_fen_move: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        by_fen4: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_fen4_move: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            fen = str(row["fen"])
            target_move = row.get("target_move_uci")
            fen4 = normalize_fen_for_eval_join(fen)
            row["fen_4field"] = fen4
            by_exact_fen[fen].append(row)
            by_fen4[fen4].append(row)
            if target_move:
                target_key = str(target_move)
                by_exact_fen_move[(fen, target_key)].append(row)
                by_fen4_move[(fen4, target_key)].append(row)
        indices[split_name] = SupervisedSplitIndex(
            split=split_name,
            rows=rows,
            by_exact_fen=dict(by_exact_fen),
            by_exact_fen_move=dict(by_exact_fen_move),
            by_fen4=dict(by_fen4),
            by_fen4_move=dict(by_fen4_move),
        )
    return indices


def _count_total_candidates(candidate_map: Mapping[str, list[dict[str, Any]]]) -> int:
    return sum(len(rows) for rows in candidate_map.values())


def _resolve_candidate_maps(
    *,
    candidate_map: Mapping[str, list[dict[str, Any]]],
    conditioned_candidate_map: Mapping[str, list[dict[str, Any]]] | None,
    base_alignment_type: str,
    base_confidence: float,
    conditioned_alignment_type: str,
    conditioned_confidence: float,
) -> tuple[ResolvedAlignment | None, str]:
    total_candidates = _count_total_candidates(candidate_map)
    if total_candidates == 0:
        return None, "no_candidates"
    if total_candidates == 1:
        split_name, rows = next(
            (split_name, rows) for split_name, rows in candidate_map.items() if rows
        )
        return (
            ResolvedAlignment(
                split=split_name,
                matched_record=rows[0],
                alignment_type=base_alignment_type,
                alignment_confidence=base_confidence,
            ),
            "matched",
        )
    if conditioned_candidate_map is not None and _count_total_candidates(conditioned_candidate_map) == 1:
        split_name, rows = next(
            (split_name, rows)
            for split_name, rows in conditioned_candidate_map.items()
            if rows
        )
        return (
            ResolvedAlignment(
                split=split_name,
                matched_record=rows[0],
                alignment_type=conditioned_alignment_type,
                alignment_confidence=conditioned_confidence,
                notes="Resolved by move-conditioned disambiguation.",
            ),
            "matched",
        )
    non_empty_splits = [split_name for split_name, rows in candidate_map.items() if rows]
    if len(non_empty_splits) > 1:
        return None, "cross_split_ambiguity"
    return None, "ambiguous_within_split"


def _source_row_id(row: Mapping[str, Any], source_name: str) -> str:
    for key in ("source_row_id", "position_id", "puzzle_id", "game_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return stable_hash_text(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False, default=str),
        prefix=f"{source_name}_",
        length=16,
    )


def _candidate_moves(row: Mapping[str, Any]) -> list[str] | None:
    raw_moves = row.get("candidate_moves")
    if raw_moves is None:
        return None
    moves = parse_space_or_comma_separated(raw_moves)
    return moves or None


def _theme_tags(row: Mapping[str, Any]) -> list[str] | None:
    tags = parse_space_or_comma_separated(row.get("theme_tags") or row.get("concept_tags") or row.get("themes"))
    return tags or None


def _build_aligned_row(
    row: Mapping[str, Any],
    *,
    source_name: str,
    match: ResolvedAlignment | None,
    notes: str | None,
) -> dict[str, Any]:
    source_row_id = _source_row_id(row, source_name)
    fen = str(row["fen"])
    matched_record = match.matched_record if match is not None else None
    alignment_notes: list[str] = []
    if match is not None and match.notes:
        alignment_notes.append(match.notes)
    if notes:
        alignment_notes.append(notes)
    payload = {
        "sidecar_id": stable_hash_text(f"{source_name}:{source_row_id}", prefix="sidecar_", length=16),
        "source": source_name,
        "source_row_id": source_row_id,
        "split": match.split if match is not None else None,
        "matched_supervised": match is not None,
        "matched_position_id": matched_record.get("position_id") if matched_record is not None else None,
        "matched_game_id": matched_record.get("game_id") if matched_record is not None else None,
        "fen": fen,
        "fen_4field": normalize_fen_for_eval_join(fen),
        "target_move_uci": row.get("target_move_uci"),
        "candidate_moves": _candidate_moves(row),
        "strategy_text": row.get("strategy_text"),
        "tactic_text": row.get("tactic_text"),
        "theme_tags": _theme_tags(row),
        "alignment_type": match.alignment_type if match is not None else "unmatched",
        "alignment_confidence": match.alignment_confidence if match is not None else 0.0,
        "notes": " ".join(alignment_notes).strip() or None,
        "preferred_move": row.get("preferred_move"),
        "history_fens": matched_record.get("history_fens") if matched_record is not None else row.get("history_fens"),
        "matched_target_move_uci": matched_record.get("target_move_uci") if matched_record is not None else None,
        "matched_next_fen": matched_record.get("next_fen") if matched_record is not None else None,
    }
    return payload


def _should_allow_move_conditioned(source_name: str, row: Mapping[str, Any], config: LanguageAlignmentConfig) -> bool:
    return bool(
        source_name == "lichess_puzzle"
        and config.allow_move_conditioned_for_puzzles
        and row.get("target_move_uci")
    )


def resolve_sidecar_alignment(
    row: Mapping[str, Any],
    *,
    source_name: str,
    supervised_indices: Mapping[str, SupervisedSplitIndex],
    config: LanguageAlignmentConfig,
) -> tuple[ResolvedAlignment | None, str]:
    """Resolve a sidecar row against train/val/test supervised indices."""
    fen = str(row["fen"])
    fen4 = normalize_fen_for_eval_join(fen)
    target_move = str(row.get("target_move_uci")) if row.get("target_move_uci") else None
    allow_move_conditioned = _should_allow_move_conditioned(source_name, row, config)

    exact_candidates = {
        split_name: split_index.by_exact_fen.get(fen, [])
        for split_name, split_index in supervised_indices.items()
    }
    exact_move_candidates = None
    if allow_move_conditioned and target_move is not None:
        exact_move_candidates = {
            split_name: split_index.by_exact_fen_move.get((fen, target_move), [])
            for split_name, split_index in supervised_indices.items()
        }
    resolved, reason = _resolve_candidate_maps(
        candidate_map=exact_candidates,
        conditioned_candidate_map=exact_move_candidates,
        base_alignment_type="fen_exact",
        base_confidence=config.exact_confidence,
        conditioned_alignment_type="fen_exact_target_move",
        conditioned_confidence=config.exact_move_confidence,
    )
    if resolved is not None or reason != "no_candidates":
        return resolved, reason

    if not config.allow_fen_4field:
        return None, "no_supervised_match"

    fen4_candidates = {
        split_name: split_index.by_fen4.get(fen4, [])
        for split_name, split_index in supervised_indices.items()
    }
    fen4_move_candidates = None
    if allow_move_conditioned and target_move is not None:
        fen4_move_candidates = {
            split_name: split_index.by_fen4_move.get((fen4, target_move), [])
            for split_name, split_index in supervised_indices.items()
        }
    resolved, reason = _resolve_candidate_maps(
        candidate_map=fen4_candidates,
        conditioned_candidate_map=fen4_move_candidates,
        base_alignment_type="fen_4field",
        base_confidence=config.fen4_confidence,
        conditioned_alignment_type="fen_4field_target_move",
        conditioned_confidence=config.fen4_move_confidence,
    )
    if resolved is not None:
        return resolved, reason
    if reason == "no_candidates":
        return None, "no_supervised_match"
    return None, reason


def _copy_optional_chessgpt_rows(
    input_path: str | Path,
    *,
    source_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_records_from_path(input_path):
        fen = row.get("fen")
        fen4 = normalize_fen_for_eval_join(str(fen)) if fen else None
        source_row_id = _source_row_id(row, source_name)
        rows.append(
            {
                "sidecar_id": stable_hash_text(f"{source_name}:{source_row_id}", prefix="sidecar_", length=16),
                "source": source_name,
                "source_row_id": source_row_id,
                "split": None,
                "matched_supervised": False,
                "matched_position_id": None,
                "matched_game_id": None,
                "fen": fen,
                "fen_4field": fen4,
                "target_move_uci": row.get("target_move_uci"),
                "candidate_moves": _candidate_moves(row),
                "strategy_text": row.get("prompt") or row.get("text"),
                "tactic_text": row.get("response"),
                "theme_tags": _theme_tags(row),
                "alignment_type": "aux_only",
                "alignment_confidence": 0.0,
                "notes": "Copied from optional ChessGPT auxiliary corpus without supervised alignment.",
                "messages": row.get("messages"),
                "schema": row.get("schema"),
            }
        )
    return rows


def build_language_alignment_index(
    *,
    supervised_train_path: str | Path,
    supervised_val_path: str | Path,
    supervised_test_path: str | Path,
    mate_path: str | Path,
    puzzle_path: str | Path,
    output_root: str | Path,
    chessgpt_text_path: str | Path | None = None,
    chessgpt_conversation_path: str | Path | None = None,
    config: LanguageAlignmentConfig | None = None,
) -> dict[str, Any]:
    """Build split-safe language-sidecar alignment outputs under ``language_v1``."""
    alignment_config = config or LanguageAlignmentConfig()
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    supervised_paths = {
        "train": supervised_train_path,
        "val": supervised_val_path,
        "test": supervised_test_path,
    }
    supervised_indices = build_supervised_indices(supervised_paths)

    source_reports: dict[str, Any] = {}

    def process_source(
        *,
        source_name: str,
        input_path: str | Path,
        matched_paths: Mapping[str, Path],
        unmatched_path: Path | None = None,
    ) -> dict[str, Any]:
        matched_by_split: dict[str, list[dict[str, Any]]] = {split_name: [] for split_name in matched_paths}
        unmatched_rows: list[dict[str, Any]] = []
        counts = Counter()
        for row in iter_records_from_path(input_path):
            counts["rows_seen"] += 1
            resolved, reason = resolve_sidecar_alignment(
                row,
                source_name=source_name,
                supervised_indices=supervised_indices,
                config=alignment_config,
            )
            if resolved is not None and resolved.split in matched_by_split:
                matched_row = _build_aligned_row(row, source_name=source_name, match=resolved, notes=None)
                matched_by_split[resolved.split].append(matched_row)
                counts["matched_rows"] += 1
                counts[f"matched_{resolved.split}"] += 1
                counts[f"alignment_{resolved.alignment_type}"] += 1
                if resolved.alignment_type.endswith("target_move"):
                    counts["move_conditioned_matches"] += 1
                continue

            if resolved is not None:
                counts["ignored_rows"] += 1
                counts[f"ignored_{resolved.split}"] += 1
                counts["ignored_due_to_output_policy"] += 1
                continue

            counts["unmatched_rows"] += 1
            counts[f"unmatched_reason_{reason}"] += 1
            if unmatched_path is not None:
                unmatched_rows.append(
                    _build_aligned_row(
                        row,
                        source_name=source_name,
                        match=None,
                        notes=f"Unmatched because {reason}.",
                    )
                )

        written_paths: dict[str, str] = {}
        for split_name, rows in matched_by_split.items():
            written_paths[split_name] = str(matched_paths[split_name])
            write_jsonl(matched_paths[split_name], rows)
        if unmatched_path is not None:
            written_paths["unmatched"] = str(unmatched_path)
            write_jsonl(unmatched_path, unmatched_rows)
        return {
            "input_path": str(input_path),
            "counts": dict(counts),
            "outputs": written_paths,
        }

    source_reports["mate"] = process_source(
        source_name="mate",
        input_path=mate_path,
        matched_paths={
            "train": output_dir / "mate_matched_train.jsonl",
            "val": output_dir / "mate_matched_val.jsonl",
            "test": output_dir / "mate_matched_test.jsonl",
        },
        unmatched_path=output_dir / "mate_unmatched.jsonl",
    )

    source_reports["puzzle"] = process_source(
        source_name="lichess_puzzle",
        input_path=puzzle_path,
        matched_paths={
            "val": output_dir / "puzzle_matched_val.jsonl",
            "test": output_dir / "puzzle_matched_test.jsonl",
        },
        unmatched_path=None,
    )

    optional_outputs: dict[str, str] = {}
    if alignment_config.copy_optional_chessgpt and chessgpt_text_path is not None and Path(chessgpt_text_path).exists():
        text_rows = _copy_optional_chessgpt_rows(chessgpt_text_path, source_name="waterhorse_chessgpt_text")
        text_output = output_dir / "chessgpt_text_aux.jsonl"
        write_jsonl(text_output, text_rows)
        optional_outputs["chessgpt_text_aux"] = str(text_output)
        source_reports["chessgpt_text"] = {
            "input_path": str(chessgpt_text_path),
            "counts": {"rows_seen": len(text_rows), "matched_rows": 0, "unmatched_rows": len(text_rows)},
            "outputs": {"aux": str(text_output)},
        }
    if (
        alignment_config.copy_optional_chessgpt
        and chessgpt_conversation_path is not None
        and Path(chessgpt_conversation_path).exists()
    ):
        conversation_rows = _copy_optional_chessgpt_rows(
            chessgpt_conversation_path,
            source_name="waterhorse_chessgpt_conversation",
        )
        conversation_output = output_dir / "chessgpt_conversation_aux.jsonl"
        write_jsonl(conversation_output, conversation_rows)
        optional_outputs["chessgpt_conversation_aux"] = str(conversation_output)
        source_reports["chessgpt_conversation"] = {
            "input_path": str(chessgpt_conversation_path),
            "counts": {
                "rows_seen": len(conversation_rows),
                "matched_rows": 0,
                "unmatched_rows": len(conversation_rows),
            },
            "outputs": {"aux": str(conversation_output)},
        }

    manifest = {
        "version": alignment_config.alignment_version,
        "config": asdict(alignment_config),
        "inputs": {
            "supervised_train": str(supervised_train_path),
            "supervised_val": str(supervised_val_path),
            "supervised_test": str(supervised_test_path),
            "mate": str(mate_path),
            "puzzle": str(puzzle_path),
            "chessgpt_text": str(chessgpt_text_path) if chessgpt_text_path is not None else None,
            "chessgpt_conversation": (
                str(chessgpt_conversation_path) if chessgpt_conversation_path is not None else None
            ),
        },
        "schema_fields": list(MATCHED_ROW_FIELDS),
        "source_reports": source_reports,
        "optional_outputs": optional_outputs,
    }
    manifest_path = manifest_dir / "alignment_manifest.yaml"
    write_yaml(manifest_path, manifest)
    return {
        "manifest_path": str(manifest_path),
        "output_root": str(output_dir),
        "source_reports": source_reports,
        "optional_outputs": optional_outputs,
    }
