"""Waterhorse/chess_data를 schema별 보조 코퍼스로 정규화한다."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import chess

from modalchess.data.preprocessing_common import (
    load_records_from_path,
    parse_space_or_comma_separated,
    stable_hash_record,
    write_jsonl,
    write_yaml,
)


@dataclass(slots=True)
class ChessGptNormalizationConfig:
    """Waterhorse corpus normalization 설정."""

    source_name: str = "waterhorse_chess_data"
    source_license: str = "Apache-2.0"
    source_version: str | None = None
    source_date: str | None = None


def _pick_first(row: Mapping[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _normalize_role(role: str) -> str:
    lowered = role.strip().lower()
    if lowered in {"human", "user", "prompt"}:
        return "user"
    if lowered in {"assistant", "gpt", "model", "bot"}:
        return "assistant"
    if lowered == "system":
        return "system"
    return lowered or "unknown"


def _normalize_messages(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        role = _normalize_role(str(item.get("role") or item.get("author") or item.get("from") or "unknown"))
        content = str(item.get("content") or item.get("text") or item.get("value") or "").strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages or None


def is_conversation_record(row: Mapping[str, Any]) -> bool:
    """row가 conversation 계열인지 판별한다."""
    for key in ("messages", "conversations", "dialogue"):
        if isinstance(row.get(key), list):
            return True
    return False


def _extract_common_chess_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    fen = _pick_first(row, ("fen", "FEN", "position"))
    if fen is not None:
        chess.Board(str(fen))
    return {
        "fen": str(fen) if fen is not None else None,
        "candidate_moves": parse_space_or_comma_separated(
            _pick_first(row, ("candidate_moves", "moves", "Moves", "legal_moves"))
        ),
    }


def normalize_text_record(
    row: Mapping[str, Any],
    config: ChessGptNormalizationConfig | None = None,
) -> dict[str, Any]:
    """text/metadata 계열 row를 정규화한다."""
    build_config = config or ChessGptNormalizationConfig()
    common = _extract_common_chess_fields(row)
    prompt = _pick_first(row, ("prompt", "instruction", "input", "question", "title"))
    response = _pick_first(row, ("response", "output", "answer"))
    text = _pick_first(row, ("text", "content", "body"))
    if prompt is None and response is None and text is None:
        raise ValueError("text row에서 핵심 텍스트 필드를 찾지 못했다.")
    return {
        "position_id": stable_hash_record(dict(row), prefix="chessgpt_text_"),
        "source": build_config.source_name,
        "schema": "text",
        **common,
        "prompt": str(prompt) if prompt is not None else None,
        "response": str(response) if response is not None else None,
        "text": str(text) if text is not None else None,
    }


def normalize_conversation_record(
    row: Mapping[str, Any],
    config: ChessGptNormalizationConfig | None = None,
) -> dict[str, Any]:
    """conversation 계열 row를 정규화한다."""
    build_config = config or ChessGptNormalizationConfig()
    common = _extract_common_chess_fields(row)
    messages = (
        _normalize_messages(row.get("messages"))
        or _normalize_messages(row.get("conversations"))
        or _normalize_messages(row.get("dialogue"))
    )
    if messages is None:
        raise ValueError("conversation row에서 messages를 정규화하지 못했다.")
    return {
        "position_id": stable_hash_record(dict(row), prefix="chessgpt_conv_"),
        "source": build_config.source_name,
        "schema": "conversation",
        **common,
        "messages": messages,
    }


def normalize_chessgpt_corpus(
    input_paths: list[str | Path],
    config: ChessGptNormalizationConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """혼합 코퍼스를 text와 conversation 파일로 분리 정규화한다."""
    build_config = config or ChessGptNormalizationConfig()
    text_records: list[dict[str, Any]] = []
    conversation_records: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "source": build_config.source_name,
        "rows_seen": 0,
        "text_rows_written": 0,
        "conversation_rows_written": 0,
        "drop_reasons": {},
    }

    def bump_drop(reason: str) -> None:
        report["drop_reasons"][reason] = int(report["drop_reasons"].get(reason, 0)) + 1

    for input_path in input_paths:
        for row in load_records_from_path(input_path):
            report["rows_seen"] += 1
            try:
                if is_conversation_record(row):
                    conversation_records.append(normalize_conversation_record(row, build_config))
                    report["conversation_rows_written"] += 1
                else:
                    text_records.append(normalize_text_record(row, build_config))
                    report["text_rows_written"] += 1
            except Exception as exc:
                bump_drop(type(exc).__name__)
    return text_records, conversation_records, report


def write_normalized_chessgpt_corpus(
    input_paths: list[str | Path],
    output_dir: str | Path,
    config: ChessGptNormalizationConfig | None = None,
) -> dict[str, Any]:
    """schema별 normalized chessgpt 코퍼스와 manifest를 기록한다."""
    build_config = config or ChessGptNormalizationConfig()
    text_records, conversation_records, report = normalize_chessgpt_corpus(input_paths, build_config)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    text_path = output_root / "chessgpt_text_corpus.jsonl"
    conversation_path = output_root / "chessgpt_conversation_corpus.jsonl"
    write_jsonl(text_path, text_records)
    write_jsonl(conversation_path, conversation_records)
    manifest = {
        "source": {
            "name": build_config.source_name,
            "license": build_config.source_license,
            "version": build_config.source_version,
            "date": build_config.source_date,
            "inputs": [str(Path(path)) for path in input_paths],
        },
        "preprocessing": asdict(build_config),
        "outputs": {
            "text_jsonl": str(text_path),
            "conversation_jsonl": str(conversation_path),
        },
        "report": report,
    }
    write_yaml(output_root / "chessgpt_manifest.yaml", manifest)
    return manifest
