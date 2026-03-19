"""MATE 데이터셋을 미래 language sidecar 형식으로 정규화한다."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import chess

from modalchess.data.preprocessing_common import (
    load_records_from_path,
    parse_space_or_comma_separated,
    stable_hash_record,
    stable_hash_text,
    write_jsonl,
    write_yaml,
)


@dataclass(slots=True)
class MateSidecarBuildConfig:
    """MATE sidecar 빌더 설정."""

    source_name: str = "mate"
    source_license: str = "MIT"
    source_version: str | None = None
    source_date: str | None = None


def _pick_first(row: Mapping[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        value = row.get(key)
        if value is not None:
            return value
    return None


def transform_mate_row(
    row: Mapping[str, Any],
    config: MateSidecarBuildConfig | None = None,
) -> dict[str, Any]:
    """MATE row를 language sidecar JSONL 레코드로 변환한다."""
    build_config = config or MateSidecarBuildConfig()
    fen = str(_pick_first(row, ("fen", "FEN", "position")) or "").strip()
    if not fen:
        raise ValueError("MATE row에 fen이 없다.")
    chess.Board(fen)

    position_seed = str(_pick_first(row, ("id", "row_id", "sample_id", "position_id")) or "")
    if position_seed:
        position_id = stable_hash_text(position_seed, prefix="mate_pos_")
    else:
        position_id = stable_hash_record(dict(row), prefix="mate_pos_")

    game_seed = str(_pick_first(row, ("game_id", "GameId", "puzzle_id")) or "")
    if game_seed:
        game_id = stable_hash_text(game_seed, prefix="mate_game_")
    else:
        game_id = stable_hash_record(dict(row), prefix="mate_game_")

    strategy_text = _pick_first(row, ("strategy_text", "strategy", "Strategy"))
    tactic_text = _pick_first(row, ("tactic_text", "tactic", "Tactic"))
    candidate_moves_raw = _pick_first(row, ("candidate_moves", "moves", "Moves", "move_candidates"))
    candidate_moves = parse_space_or_comma_separated(candidate_moves_raw)

    return {
        "position_id": position_id,
        "game_id": game_id,
        "source": build_config.source_name,
        "fen": fen,
        "candidate_moves": candidate_moves,
        "strategy_text": str(strategy_text) if strategy_text is not None else None,
        "tactic_text": str(tactic_text) if tactic_text is not None else None,
    }


def build_mate_sidecar_records(
    input_paths: list[str | Path],
    config: MateSidecarBuildConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """MATE 원천 파일들을 sidecar 레코드로 정리한다."""
    build_config = config or MateSidecarBuildConfig()
    records: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "source": build_config.source_name,
        "rows_seen": 0,
        "rows_written": 0,
        "drop_reasons": {},
    }

    def bump_drop(reason: str) -> None:
        report["drop_reasons"][reason] = int(report["drop_reasons"].get(reason, 0)) + 1

    for input_path in input_paths:
        for row in load_records_from_path(input_path):
            report["rows_seen"] += 1
            try:
                records.append(transform_mate_row(row, build_config))
                report["rows_written"] += 1
            except Exception as exc:
                bump_drop(type(exc).__name__)
    return records, report


def write_mate_sidecar(
    input_paths: list[str | Path],
    output_path: str | Path,
    config: MateSidecarBuildConfig | None = None,
) -> dict[str, Any]:
    """MATE sidecar JSONL과 manifest를 기록한다."""
    build_config = config or MateSidecarBuildConfig()
    records, report = build_mate_sidecar_records(input_paths, build_config)
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
    write_yaml(output_file.parent / "mate_manifest.yaml", manifest)
    return manifest
