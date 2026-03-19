"""MATE 데이터셋을 미래 language sidecar 형식으로 정규화한다."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
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


FEN_IN_TEXT_PATTERN = re.compile(r'FEN[^"]*"([^"]+)"')
MOVE_PATTERN = re.compile(r"Move([A-Z]):([a-h][1-8][a-h][1-8][nbrq]?)")


def _extract_fen_from_prompt_text(text: str) -> str | None:
    match = FEN_IN_TEXT_PATTERN.search(text)
    return match.group(1).strip() if match else None


def _extract_candidate_moves_from_prompt_text(text: str) -> list[str]:
    moves: list[str] = []
    for _, move in MOVE_PATTERN.findall(text):
        if move not in moves:
            moves.append(move)
    return moves


def _segment_between(text: str, start_marker: str, end_markers: tuple[str, ...]) -> str | None:
    start_index = text.find(start_marker)
    if start_index < 0:
        return None
    start_index += len(start_marker)
    end_index = len(text)
    for end_marker in end_markers:
        candidate_index = text.find(end_marker, start_index)
        if candidate_index >= 0:
            end_index = min(end_index, candidate_index)
    segment = text[start_index:end_index].strip()
    return segment or None


def _extract_strategy_tactic_texts(text: str) -> tuple[str | None, str | None]:
    strategies: list[str] = []
    tactics: list[str] = []
    for label in ("A", "B", "C", "D"):
        move_token = f"Move{label}:"
        tactic_token = f"Tactic{label}:"
        strategy = _segment_between(text, move_token, (tactic_token, f"Move{chr(ord(label) + 1)}:", "Tactic"))
        if strategy is not None:
            comma_index = strategy.find(",")
            if comma_index >= 0:
                strategy = strategy[comma_index + 1 :].strip()
            strategies.append(f"{label}: {strategy}")
        tactic = _segment_between(text, tactic_token, (f"Move{chr(ord(label) + 1)}:", f"Tactic{chr(ord(label) + 1)}:"))
        if tactic is not None:
            tactics.append(f"{label}: {tactic}")
    strategy_text = " || ".join(strategies) if strategies else None
    tactic_text = " || ".join(tactics) if tactics else None
    return strategy_text, tactic_text


def transform_mate_row(
    row: Mapping[str, Any],
    config: MateSidecarBuildConfig | None = None,
) -> dict[str, Any]:
    """MATE row를 language sidecar JSONL 레코드로 변환한다."""
    build_config = config or MateSidecarBuildConfig()
    prompt_input = str(_pick_first(row, ("input", "prompt", "instruction")) or "").strip()
    fen = str(_pick_first(row, ("fen", "FEN", "position")) or "").strip()
    if not fen and prompt_input:
        extracted_fen = _extract_fen_from_prompt_text(prompt_input)
        fen = extracted_fen or ""
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
    if not candidate_moves and prompt_input:
        candidate_moves = _extract_candidate_moves_from_prompt_text(prompt_input)
    if (strategy_text is None or tactic_text is None) and prompt_input:
        prompt_strategy_text, prompt_tactic_text = _extract_strategy_tactic_texts(prompt_input)
        strategy_text = strategy_text or prompt_strategy_text
        tactic_text = tactic_text or prompt_tactic_text

    return {
        "position_id": position_id,
        "game_id": game_id,
        "source": build_config.source_name,
        "fen": fen,
        "candidate_moves": candidate_moves,
        "strategy_text": str(strategy_text) if strategy_text is not None else None,
        "tactic_text": str(tactic_text) if tactic_text is not None else None,
        "preferred_move": str(_pick_first(row, ("output", "label_move", "best_move"))).strip()
        if _pick_first(row, ("output", "label_move", "best_move")) is not None
        else None,
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
