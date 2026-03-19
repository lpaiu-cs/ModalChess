"""Lichess puzzle 원천을 ModalChess sidecar JSONL로 변환한다."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chess

from modalchess.data.preprocessing_common import (
    StableSplitConfig,
    assign_split_by_game_id,
    load_records_from_path,
    parse_space_or_comma_separated,
    special_rule_flags,
    validate_modalchess_record,
    write_jsonl,
    write_yaml,
)


@dataclass(slots=True)
class PuzzleSidecarBuildConfig:
    """Lichess puzzle sidecar 빌더 설정."""

    source_name: str = "lichess_puzzle"
    source_license: str = "CC0"
    source_version: str | None = None
    source_date: str | None = None
    include_history: bool = True
    emit_legal_moves: bool = False
    assign_split: bool = False
    split_config: StableSplitConfig = field(default_factory=StableSplitConfig)


def transform_puzzle_row(
    row: dict[str, Any],
    config: PuzzleSidecarBuildConfig | None = None,
) -> dict[str, Any] | None:
    """Lichess puzzle 한 행을 ModalChess JSONL 레코드로 바꾼다."""
    build_config = config or PuzzleSidecarBuildConfig()
    puzzle_id = str(row.get("PuzzleId") or row.get("puzzle_id") or "").strip()
    if not puzzle_id:
        raise ValueError("PuzzleId가 누락됐다.")
    initial_fen = str(row.get("FEN") or row.get("fen") or "").strip()
    if not initial_fen:
        raise ValueError(f"FEN이 누락됐다: {puzzle_id}")
    moves = parse_space_or_comma_separated(row.get("Moves") or row.get("moves"))
    if len(moves) < 2:
        return None

    board = chess.Board(initial_fen)
    first_move = chess.Move.from_uci(moves[0])
    if first_move not in board.legal_moves:
        raise ValueError(f"첫 puzzle move가 합법 수가 아니다: {puzzle_id} / {moves[0]}")
    board.push(first_move)
    current_fen = board.fen(en_passant="fen")

    target_move = chess.Move.from_uci(moves[1])
    if target_move not in board.legal_moves:
        raise ValueError(f"두 번째 puzzle move가 합법 수가 아니다: {puzzle_id} / {moves[1]}")
    next_board = board.copy(stack=False)
    next_board.push(target_move)

    game_id = str(row.get("GameId") or row.get("game_id") or puzzle_id)
    record: dict[str, Any] = {
        "position_id": puzzle_id,
        "game_id": game_id,
        "fen": current_fen,
        "target_move_uci": target_move.uci(),
        "next_fen": next_board.fen(en_passant="fen"),
        "concept_tags": parse_space_or_comma_separated(row.get("Themes") or row.get("themes")),
        "source": build_config.source_name,
    }
    if build_config.include_history:
        record["history_fens"] = [current_fen]
    if build_config.emit_legal_moves:
        record["legal_moves_uci"] = [move.uci() for move in board.legal_moves]
    if build_config.assign_split:
        record["split"] = assign_split_by_game_id(game_id, build_config.split_config)
    validate_modalchess_record(record, require_target_move=True)
    return record


def build_puzzle_records(
    input_paths: list[str | Path],
    config: PuzzleSidecarBuildConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """여러 puzzle 원천 파일을 sidecar 레코드로 변환한다."""
    build_config = config or PuzzleSidecarBuildConfig()
    records: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "source": build_config.source_name,
        "rows_seen": 0,
        "rows_written": 0,
        "drop_reasons": {},
        "subset_counts": {"promotion": 0, "castling": 0, "en_passant": 0, "check_evasion": 0},
        "split_counts": {"train": 0, "val": 0, "test": 0},
    }

    def bump_drop(reason: str) -> None:
        report["drop_reasons"][reason] = int(report["drop_reasons"].get(reason, 0)) + 1

    for input_path in input_paths:
        for row in load_records_from_path(input_path):
            report["rows_seen"] += 1
            try:
                record = transform_puzzle_row(row, build_config)
            except Exception as exc:
                bump_drop(type(exc).__name__)
                continue
            if record is None:
                bump_drop("insufficient_moves")
                continue
            records.append(record)
            flags = special_rule_flags(str(record["fen"]), str(record["target_move_uci"]))
            for key, active in flags.items():
                report["subset_counts"][key] += int(active)
            if "split" in record:
                report["split_counts"][str(record["split"])] += 1
            report["rows_written"] += 1
    return records, report


def write_puzzle_sidecar(
    input_paths: list[str | Path],
    output_path: str | Path,
    config: PuzzleSidecarBuildConfig | None = None,
) -> dict[str, Any]:
    """puzzle sidecar JSONL과 manifest를 기록한다."""
    build_config = config or PuzzleSidecarBuildConfig()
    records, report = build_puzzle_records(input_paths, build_config)
    output_file = Path(output_path)
    write_jsonl(output_file, records)
    manifest = {
        "source": {
            "name": build_config.source_name,
            "license": build_config.source_license,
            "version": build_config.source_version,
            "date": build_config.source_date,
            "inputs": [str(Path(path)) for path in input_paths],
        },
        "preprocessing": asdict(build_config),
        "outputs": {"jsonl": str(output_file)},
        "report": report,
    }
    write_yaml(output_file.parent / "puzzle_manifest.yaml", manifest)
    return manifest
