"""PGN 원천을 ModalChess supervised JSONL로 변환하는 빌더."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from modalchess.data.preprocessing_common import (
    StableSplitConfig,
    assign_split_by_game_id,
    open_text_input,
    special_rule_flags,
    stable_hash_text,
    validate_modalchess_record,
    write_jsonl,
    write_yaml,
)


@dataclass(slots=True)
class PgnPilotBuildConfig:
    """PGN supervised 파일럿 빌더 설정."""

    source_name: str = "lichess_pgn"
    source_license: str = "CC0"
    source_version: str | None = None
    source_date: str | None = None
    standard_only: bool = True
    rated_only: bool = False
    include_history: bool = True
    emit_legal_moves: bool = False
    min_game_plies: int = 1
    max_game_plies: int | None = None
    min_ply_index: int = 0
    max_ply_index: int | None = None
    explicit_split_header: str = "Split"
    split_config: StableSplitConfig = field(default_factory=StableSplitConfig)


def derive_pgn_game_id(headers: dict[str, str]) -> str:
    """PGN 헤더에서 안정적인 game_id를 만든다."""
    site = (headers.get("Site") or "").strip()
    if site and site != "?":
        return stable_hash_text(site, prefix="game_")
    fallback = "|".join(
        [
            (headers.get("Event") or "?").strip(),
            (headers.get("Round") or "?").strip(),
            (headers.get("White") or "?").strip(),
            (headers.get("Black") or "?").strip(),
            (headers.get("Date") or "?").strip(),
        ]
    )
    return stable_hash_text(fallback, prefix="game_")


def _is_standard_game(headers: dict[str, str]) -> bool:
    variant = (headers.get("Variant") or "Standard").strip().lower()
    return variant in {"", "standard", "chess", "from position"}


def _is_rated_game(headers: dict[str, str]) -> bool:
    rated_header = (headers.get("Rated") or "").strip().lower()
    if rated_header in {"true", "yes", "1"}:
        return True
    event = (headers.get("Event") or "").strip().lower()
    return "rated" in event


def _resolve_game_split(headers: dict[str, str], game_id: str, config: PgnPilotBuildConfig) -> str:
    explicit_split = (headers.get(config.explicit_split_header) or "").strip().lower()
    if explicit_split in {"train", "val", "test"}:
        return explicit_split
    return assign_split_by_game_id(game_id, config.split_config)


def _base_record_from_position(
    *,
    game_id: str,
    ply_index: int,
    fen: str,
    history_fens: list[str] | None,
    target_move_uci: str,
    next_fen: str,
    split_name: str,
    board: chess.Board,
    config: PgnPilotBuildConfig,
) -> dict[str, Any]:
    position_id = stable_hash_text(f"{game_id}:{ply_index}", prefix="pos_")
    record: dict[str, Any] = {
        "position_id": position_id,
        "game_id": game_id,
        "fen": fen,
        "target_move_uci": target_move_uci,
        "next_fen": next_fen,
        "split": split_name,
    }
    if history_fens is not None:
        record["history_fens"] = history_fens
    if config.emit_legal_moves:
        record["legal_moves_uci"] = [move.uci() for move in board.legal_moves]
    return record


def build_supervised_records_from_pgn(
    input_paths: list[str | Path],
    config: PgnPilotBuildConfig | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """PGN 파일들을 supervised train/val/test 레코드로 변환한다."""
    build_config = config or PgnPilotBuildConfig()
    records_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    subset_counts: dict[str, dict[str, int]] = {
        "train": {"promotion": 0, "castling": 0, "en_passant": 0, "check_evasion": 0},
        "val": {"promotion": 0, "castling": 0, "en_passant": 0, "check_evasion": 0},
        "test": {"promotion": 0, "castling": 0, "en_passant": 0, "check_evasion": 0},
    }
    report: dict[str, Any] = {
        "source": build_config.source_name,
        "games_seen": 0,
        "games_kept": 0,
        "positions_written": 0,
        "drop_reasons": {},
        "positions_skipped_outside_ply_range": 0,
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "subset_counts": subset_counts,
        "source_counts": {build_config.source_name: 0},
    }

    def bump_drop(reason: str) -> None:
        report["drop_reasons"][reason] = int(report["drop_reasons"].get(reason, 0)) + 1

    for input_path in input_paths:
        path = Path(input_path)
        with open_text_input(path) as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                report["games_seen"] += 1
                headers = dict(game.headers)

                if build_config.standard_only and not _is_standard_game(headers):
                    bump_drop("non_standard_variant")
                    continue
                if build_config.rated_only and not _is_rated_game(headers):
                    bump_drop("unrated_game")
                    continue

                moves = list(game.mainline_moves())
                if len(moves) < build_config.min_game_plies:
                    bump_drop("too_short_game")
                    continue
                if build_config.max_game_plies is not None and len(moves) > build_config.max_game_plies:
                    bump_drop("too_long_game")
                    continue

                board = game.board()
                game_id = derive_pgn_game_id(headers)
                split_name = _resolve_game_split(headers, game_id, build_config)
                history = [board.fen(en_passant="fen")]
                report["games_kept"] += 1

                for ply_index, move in enumerate(moves):
                    current_fen = board.fen(en_passant="fen")
                    within_min = ply_index >= build_config.min_ply_index
                    within_max = (
                        True
                        if build_config.max_ply_index is None
                        else ply_index <= build_config.max_ply_index
                    )
                    next_board = board.copy(stack=False)
                    next_board.push(move)
                    next_fen = next_board.fen(en_passant="fen")
                    if within_min and within_max:
                        history_snapshot = list(history) if build_config.include_history else None
                        record = _base_record_from_position(
                            game_id=game_id,
                            ply_index=ply_index,
                            fen=current_fen,
                            history_fens=history_snapshot,
                            target_move_uci=move.uci(),
                            next_fen=next_fen,
                            split_name=split_name,
                            board=board,
                            config=build_config,
                        )
                        validate_modalchess_record(record, require_target_move=True)
                        records_by_split[split_name].append(record)
                        flags = special_rule_flags(current_fen, move.uci())
                        for key, active in flags.items():
                            subset_counts[split_name][key] += int(active)
                        report["split_counts"][split_name] += 1
                        report["positions_written"] += 1
                        report["source_counts"][build_config.source_name] += 1
                    else:
                        report["positions_skipped_outside_ply_range"] += 1
                    board.push(move)
                    history.append(board.fen(en_passant="fen"))
    return records_by_split, report


def write_supervised_pilot_from_pgn(
    input_paths: list[str | Path],
    output_dir: str | Path,
    config: PgnPilotBuildConfig | None = None,
) -> dict[str, Any]:
    """PGN 원천을 ModalChess supervised split 파일로 기록한다."""
    build_config = config or PgnPilotBuildConfig()
    records_by_split, report = build_supervised_records_from_pgn(input_paths, build_config)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    output_files = {
        "train": output_root / "supervised_train.jsonl",
        "val": output_root / "supervised_val.jsonl",
        "test": output_root / "supervised_test.jsonl",
    }
    for split_name, records in records_by_split.items():
        write_jsonl(output_files[split_name], records)

    manifest = {
        "source": {
            "name": build_config.source_name,
            "license": build_config.source_license,
            "version": build_config.source_version,
            "date": build_config.source_date,
            "inputs": [str(Path(path)) for path in input_paths],
        },
        "preprocessing": asdict(build_config),
        "outputs": {name: str(path) for name, path in output_files.items()},
        "report": report,
    }
    write_yaml(output_root / "pgn_manifest.yaml", manifest)
    return manifest
