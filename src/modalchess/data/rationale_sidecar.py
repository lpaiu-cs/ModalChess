"""Week-4 rationale-ready sidecar builders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import chess

from modalchess.data.preprocessing_common import (
    iter_records_from_path,
    special_rule_flags,
    stable_hash_text,
    write_jsonl,
    write_yaml,
)


@dataclass(slots=True)
class RationaleBuildConfig:
    """Configuration for rationale-ready sidecar export."""

    rationale_version: str = "week4_rationale_v1"
    max_rationale_chars: int = 240


def _split_option_text(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    parts = [part.strip() for part in value.split("||") if part.strip()]
    options: dict[str, str] = {}
    for part in parts:
        if ":" not in part:
            continue
        key, text = part.split(":", 1)
        options[key.strip()] = text.strip()
    return options


def _extract_preferred_letter(preferred_move: str | None) -> str | None:
    if not preferred_move:
        return None
    if preferred_move.startswith("Move") and ":" in preferred_move:
        return preferred_move[4:].split(":", 1)[0]
    return None


def _clip_text(text: str, max_chars: int) -> str:
    clipped = " ".join(text.split())
    if len(clipped) <= max_chars:
        return clipped
    return clipped[: max_chars - 3].rstrip() + "..."


def _coarse_text_tags(text: str) -> tuple[list[str], list[str], list[str]]:
    lowered = text.lower()
    motif_tags: list[str] = []
    threat_tags: list[str] = []
    defense_tags: list[str] = []
    if "checkmate" in lowered or "mate" in lowered:
        motif_tags.append("checkmate_pattern")
        threat_tags.append("mate_threat")
    if "check" in lowered:
        motif_tags.append("check")
        threat_tags.append("forcing_check")
    if "trade" in lowered or "exchange" in lowered:
        motif_tags.append("exchange")
    if "open file" in lowered or "open diagonal" in lowered:
        motif_tags.append("line_opening")
    if "king" in lowered:
        threat_tags.append("king_pressure")
    if "defend" in lowered or "respond" in lowered or "legal king move" in lowered:
        defense_tags.append("defensive_response")
    return motif_tags, threat_tags, defense_tags


def _focus_fields(fen: str, target_move_uci: str | None) -> tuple[list[str], list[str], list[str], list[str], bool]:
    board = chess.Board(fen)
    focus_squares: list[str] = []
    focus_pieces: list[str] = []
    threat_tags: list[str] = []
    defense_tags: list[str] = []
    king_safety_flag = board.is_check()
    if target_move_uci is None:
        if king_safety_flag:
            king_square = board.king(board.turn)
            if king_square is not None:
                focus_squares.append(chess.square_name(king_square))
                focus_pieces.append("king")
                defense_tags.append("check_evasion")
        return focus_squares, focus_pieces, threat_tags, defense_tags, king_safety_flag

    move = chess.Move.from_uci(target_move_uci)
    src_name = chess.square_name(move.from_square)
    dst_name = chess.square_name(move.to_square)
    focus_squares.extend([src_name, dst_name])
    moving_piece = board.piece_at(move.from_square)
    captured_piece = board.piece_at(move.to_square)
    if moving_piece is not None:
        focus_pieces.append(moving_piece.symbol())
    if captured_piece is not None:
        focus_pieces.append(captured_piece.symbol())
        threat_tags.append("capture")
    next_board = board.copy(stack=False)
    next_board.push(move)
    if next_board.is_check():
        threat_tags.append("gives_check")
    if board.is_check():
        defense_tags.append("check_evasion")
    return focus_squares, focus_pieces, threat_tags, defense_tags, king_safety_flag


def _motif_tags(row: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    motif_tags = list(row.get("theme_tags") or [])
    threat_tags: list[str] = []
    defense_tags: list[str] = []
    text_chunks = [
        str(row.get("strategy_text") or ""),
        str(row.get("tactic_text") or ""),
    ]
    for text in text_chunks:
        text_motifs, text_threats, text_defenses = _coarse_text_tags(text)
        motif_tags.extend(text_motifs)
        threat_tags.extend(text_threats)
        defense_tags.extend(text_defenses)
    return sorted(set(motif_tags)), sorted(set(threat_tags)), sorted(set(defense_tags))


def _build_rationale_short(row: dict[str, Any], config: RationaleBuildConfig) -> str:
    preferred_letter = _extract_preferred_letter(row.get("preferred_move"))
    selected_segments: list[str] = []
    if row.get("source") == "mate":
        strategy_options = _split_option_text(row.get("strategy_text"))
        tactic_options = _split_option_text(row.get("tactic_text"))
        if preferred_letter and preferred_letter in strategy_options:
            selected_segments.append(strategy_options[preferred_letter])
        elif strategy_options:
            selected_segments.append(next(iter(strategy_options.values())))
        elif row.get("strategy_text"):
            selected_segments.append(str(row["strategy_text"]))
        if preferred_letter and preferred_letter in tactic_options:
            selected_segments.append(tactic_options[preferred_letter])
        elif tactic_options:
            selected_segments.append(next(iter(tactic_options.values())))
        elif row.get("tactic_text"):
            selected_segments.append(str(row["tactic_text"]))
    elif row.get("theme_tags"):
        selected_segments.append("Puzzle themes: " + ", ".join(str(tag) for tag in row["theme_tags"]))
    elif row.get("strategy_text") or row.get("tactic_text"):
        if row.get("strategy_text"):
            selected_segments.append(str(row["strategy_text"]))
        if row.get("tactic_text"):
            selected_segments.append(str(row["tactic_text"]))
    if not selected_segments and row.get("matched_target_move_uci"):
        selected_segments.append(f"Move focus: {row['matched_target_move_uci']}")
    return _clip_text(" ".join(selected_segments).strip(), config.max_rationale_chars)


def _rationale_row(row: dict[str, Any], config: RationaleBuildConfig) -> dict[str, Any]:
    position_id = str(row["matched_position_id"])
    target_move_uci = row.get("matched_target_move_uci") or row.get("target_move_uci")
    flags = special_rule_flags(str(row["fen"]), str(target_move_uci) if target_move_uci else None)
    focus_squares, focus_pieces, move_threat_tags, move_defense_tags, king_safety_flag = _focus_fields(
        str(row["fen"]),
        str(target_move_uci) if target_move_uci else None,
    )
    motif_tags, text_threat_tags, text_defense_tags = _motif_tags(row)
    threat_tags = sorted(set(move_threat_tags + text_threat_tags))
    defense_tags = sorted(set(move_defense_tags + text_defense_tags))
    return {
        "rationale_id": stable_hash_text(f"{row['sidecar_id']}:{position_id}", prefix="rationale_", length=16),
        "position_id": position_id,
        "matched_game_id": row.get("matched_game_id"),
        "fen": row["fen"],
        "target_move_uci": target_move_uci,
        "focus_squares": focus_squares,
        "focus_pieces": focus_pieces,
        "motif_tags": motif_tags,
        "threat_tags": threat_tags,
        "defense_tags": defense_tags,
        "king_safety_flag": king_safety_flag,
        "promotion_flag": bool(flags["promotion"]),
        "castling_flag": bool(flags["castling"]),
        "en_passant_flag": bool(flags["en_passant"]),
        "check_evasion_flag": bool(flags["check_evasion"]),
        "rationale_short": _build_rationale_short(row, config),
        "source": row["source"],
        "source_confidence": float(row.get("alignment_confidence", 0.0)),
        "split": row.get("split"),
        "sidecar_id": row.get("sidecar_id"),
        "source_row_id": row.get("source_row_id"),
        "alignment_type": row.get("alignment_type"),
        "theme_tags": row.get("theme_tags"),
    }


def _load_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        path_obj = Path(path)
        if not path_obj.exists():
            continue
        rows.extend(dict(row) for row in iter_records_from_path(path_obj))
    return rows


def build_rationale_sidecars(
    *,
    input_root: str | Path,
    output_root: str | Path,
    config: RationaleBuildConfig | None = None,
) -> dict[str, Any]:
    """Build rationale-ready train/val/test JSONL files from matched sidecars."""
    rationale_config = config or RationaleBuildConfig()
    input_dir = Path(input_root)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_inputs = {
        "train": [
            input_dir / "mate_matched_train.jsonl",
            input_dir / "puzzle_matched_train.jsonl",
        ],
        "val": [
            input_dir / "mate_matched_val.jsonl",
            input_dir / "puzzle_matched_val.jsonl",
        ],
        "test": [
            input_dir / "mate_matched_test.jsonl",
            input_dir / "puzzle_matched_test.jsonl",
        ],
    }

    report: dict[str, Any] = {
        "version": rationale_config.rationale_version,
        "splits": {},
    }
    output_paths: dict[str, str] = {}
    for split_name, paths in split_inputs.items():
        rows = [row for row in _load_rows(paths) if row.get("matched_supervised")]
        rationale_rows = [_rationale_row(row, rationale_config) for row in rows]
        output_path = output_dir / f"rationale_{split_name}.jsonl"
        write_jsonl(output_path, rationale_rows)
        output_paths[split_name] = str(output_path)
        report["splits"][split_name] = {
            "row_count": len(rationale_rows),
            "sources": sorted({str(row["source"]) for row in rationale_rows}),
        }

    manifest_path = output_dir / "manifests" / "rationale_manifest.yaml"
    write_yaml(
        manifest_path,
        {
            "config": asdict(rationale_config),
            "inputs": {split_name: [str(path) for path in paths] for split_name, paths in split_inputs.items()},
            "outputs": output_paths,
            "report": report,
        },
    )
    return {
        "outputs": output_paths,
        "manifest_path": str(manifest_path),
        "report": report,
    }
