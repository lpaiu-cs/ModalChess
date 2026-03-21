"""Auxiliary language source audit and normalization for week-7/week-8."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import chess
import chess.pgn

from modalchess.data.preprocessing_common import (
    StableSplitConfig,
    assign_split_by_game_id,
    iter_records_from_path,
    normalize_fen_for_eval_join,
    parse_space_or_comma_separated,
    stable_hash_record,
    stable_hash_text,
    write_jsonl,
    write_yaml,
)


DEFAULT_AUX_SOURCE_PATHS = {
    "waterhorse_raw": Path("data/pilot/raw/hf/waterhorse_chess_data"),
    "chessgpt_text_sample": Path("data/pilot/samples/chessgpt_text_corpus.jsonl"),
    "chessgpt_conversation_sample": Path("data/pilot/samples/chessgpt_conversation_corpus.jsonl"),
}


@dataclass(slots=True)
class AuxLanguageBuildConfig:
    """Auxiliary language corpus build configuration."""

    split_config: StableSplitConfig = field(
        default_factory=lambda: StableSplitConfig(salt="modalchess_week8_aux")
    )


def _is_structured_row_file(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".metadata"):
        return False
    if any(part == ".cache" for part in path.parts):
        return False
    return (
        name.endswith(".jsonl")
        or name.endswith(".json")
        or name.endswith(".csv")
        or ".jsonl-" in name
    )


def _iter_source_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    files = [file_path for file_path in path.rglob("*") if file_path.is_file() and _is_structured_row_file(file_path)]
    return sorted(files)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if ".jsonl-" in path.name.lower():
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    try:
        return [dict(row) for row in iter_records_from_path(path)]
    except Exception:
        return []


def _flatten_text_row(row: Mapping[str, Any]) -> str | None:
    segments = [
        str(row.get("prompt") or "").strip(),
        str(row.get("response") or "").strip(),
        str(row.get("text") or "").strip(),
    ]
    text = "\n".join(segment for segment in segments if segment)
    return text or None


def _flatten_conversation_row(row: Mapping[str, Any]) -> tuple[str | None, list[dict[str, str]] | None]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None, None
    normalized_messages: list[dict[str, str]] = []
    lines: list[str] = []
    for item in messages:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "unknown").strip()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        normalized_messages.append({"role": role, "content": content})
        lines.append(f"{role}: {content}")
    if not normalized_messages:
        return None, None
    return "\n".join(lines), normalized_messages


def _natural_text_and_messages(row: Mapping[str, Any]) -> tuple[str | None, list[dict[str, str]] | None]:
    schema = str(row.get("schema") or "")
    if schema == "conversation":
        return _flatten_conversation_row(row)
    return _flatten_text_row(row), None


def _extract_pgn_anchor(text: str | None) -> dict[str, Any] | None:
    if not text or "[" not in text:
        return None
    try:
        game = chess.pgn.read_game(io.StringIO(text))
    except Exception:
        return None
    if game is None:
        return None
    try:
        board = game.board()
        fen = board.fen(en_passant="fen")
    except Exception:
        return None
    try:
        chess.Board(fen)
    except ValueError:
        return None
    candidate_moves = [move.uci() for move in game.mainline_moves()]
    return {
        "fen": fen,
        "candidate_moves": candidate_moves,
        "board_anchor_type": "pgn_mainline",
    }


def _extract_row_anchor(row: Mapping[str, Any], text: str | None) -> dict[str, Any] | None:
    fen = row.get("fen")
    if fen not in (None, ""):
        fen_text = str(fen)
        return {
            "fen": fen_text,
            "candidate_moves": parse_space_or_comma_separated(row.get("candidate_moves")),
            "board_anchor_type": "fen",
        }
    return _extract_pgn_anchor(text)


def _classify_row(row: Mapping[str, Any]) -> tuple[str, str | None, list[dict[str, str]] | None, dict[str, Any] | None]:
    text, messages = _natural_text_and_messages(row)
    anchor = _extract_row_anchor(row, text)
    if text and anchor is not None:
        return "board_anchored", text, messages, anchor
    if text:
        return "text_only", text, messages, None
    return "unusable", text, messages, None


def _source_row_id(row: Mapping[str, Any]) -> str:
    for key in ("pipeline_key", "position_id", "source_row_id", "id", "game_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return stable_hash_record(row, prefix="row_", length=16)


def _source_schema(row: Mapping[str, Any], source_name: str) -> str:
    if row.get("schema") not in (None, ""):
        return str(row["schema"])
    if source_name == "waterhorse_raw":
        return "waterhorse_jsonl"
    return "unknown"


def _source_group_id(row: Mapping[str, Any], source_row_id: str, anchor: dict[str, Any] | None) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        game_id = metadata.get("game_id")
        if game_id not in (None, ""):
            return str(game_id)
    if anchor is not None and anchor.get("board_anchor_type") == "pgn_mainline":
        text = str(row.get("text") or "")
        return stable_hash_text(text[:2048], prefix="pgn_", length=16)
    return source_row_id


def _normalize_row(
    *,
    source_name: str,
    source_file: Path,
    row: Mapping[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    category, text, messages, anchor = _classify_row(row)
    if category == "unusable":
        return category, None
    source_row_id = _source_row_id(row)
    payload = {
        "probe_id": stable_hash_text(f"{source_name}:{source_row_id}", prefix="aux_", length=16),
        "aux_id": stable_hash_text(f"{source_name}:{source_row_id}", prefix="aux_", length=16),
        "source": str(row.get("source") or source_name),
        "source_file": str(source_file),
        "source_schema": _source_schema(row, source_name),
        "source_row_id": source_row_id,
        "fen": anchor["fen"] if anchor is not None else None,
        "fen_4field": normalize_fen_for_eval_join(str(anchor["fen"])) if anchor is not None else None,
        "candidate_moves": list(anchor["candidate_moves"]) if anchor is not None else parse_space_or_comma_separated(row.get("candidate_moves")),
        "text": text,
        "messages": messages,
        "board_anchor_type": anchor["board_anchor_type"] if anchor is not None else None,
        "text_kind": "conversation" if messages is not None else "natural_text",
        "metadata": {
            "metadata": row.get("metadata"),
            "pipeline_key": row.get("pipeline_key"),
            "prompt": row.get("prompt"),
            "response": row.get("response"),
        },
    }
    if category == "board_anchored":
        payload["split_group_id"] = _source_group_id(row, source_row_id, anchor)
    return category, payload


def _file_report(source_name: str, file_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _load_rows(file_path)
    board_anchored_rows: list[dict[str, Any]] = []
    text_only_rows: list[dict[str, Any]] = []
    natural_text_rows = 0
    explicit_fen_rows = 0
    explicit_move_list_rows = 0
    recoverable_board_anchor_rows = 0
    conversation_rows = 0
    unusable_rows = 0

    for row in rows:
        category, text, messages, anchor = _classify_row(row)
        natural_text_rows += int(bool(text))
        explicit_fen_rows += int(row.get("fen") not in (None, ""))
        explicit_move_list_rows += int(bool(parse_space_or_comma_separated(row.get("candidate_moves"))))
        recoverable_board_anchor_rows += int(anchor is not None)
        conversation_rows += int(messages is not None)
        if category == "board_anchored":
            normalized_category, payload = _normalize_row(source_name=source_name, source_file=file_path, row=row)
            if normalized_category == "board_anchored" and payload is not None:
                board_anchored_rows.append(payload)
        elif category == "text_only":
            normalized_category, payload = _normalize_row(source_name=source_name, source_file=file_path, row=row)
            if normalized_category == "text_only" and payload is not None:
                text_only_rows.append(payload)
        else:
            unusable_rows += 1

    schema_hint = str(rows[0].get("schema")) if rows and rows[0].get("schema") not in (None, "") else "unknown"
    report = {
        "path": str(file_path),
        "row_count": len(rows),
        "schema_hint": schema_hint,
        "sample_fields": sorted(rows[0].keys()) if rows else [],
        "explicit_fen_rows": explicit_fen_rows,
        "explicit_move_list_rows": explicit_move_list_rows,
        "recoverable_board_anchor_rows": recoverable_board_anchor_rows,
        "natural_text_rows": natural_text_rows,
        "conversation_rows": conversation_rows,
        "board_anchored_rows": len(board_anchored_rows),
        "text_only_rows": len(text_only_rows),
        "unusable_rows": unusable_rows,
    }
    return report, board_anchored_rows, text_only_rows


def _collect_source_inventory(source_paths: Mapping[str, str | Path]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    resolved_paths = {name: Path(path) for name, path in source_paths.items()}
    source_reports: dict[str, Any] = {}
    board_anchored_rows: list[dict[str, Any]] = []
    text_only_rows: list[dict[str, Any]] = []
    overall_counts = Counter()

    for source_name, path in resolved_paths.items():
        if not path.exists():
            source_reports[source_name] = {
                "path": str(path),
                "available": False,
                "is_file": False,
                "row_count": 0,
                "structured_file_count": 0,
                "files": [],
                "explicit_fen_rows": 0,
                "explicit_move_list_rows": 0,
                "recoverable_board_anchor_rows": 0,
                "natural_text_rows": 0,
                "conversation_rows": 0,
                "board_anchored_rows": 0,
                "text_only_rows": 0,
                "unusable_rows": 0,
            }
            continue

        source_file_reports: list[dict[str, Any]] = []
        source_counts = Counter()
        for file_path in _iter_source_files(path):
            file_report, file_board_rows, file_text_rows = _file_report(source_name, file_path)
            source_file_reports.append(file_report)
            board_anchored_rows.extend(file_board_rows)
            text_only_rows.extend(file_text_rows)
            source_counts["row_count"] += int(file_report["row_count"])
            source_counts["explicit_fen_rows"] += int(file_report["explicit_fen_rows"])
            source_counts["explicit_move_list_rows"] += int(file_report["explicit_move_list_rows"])
            source_counts["recoverable_board_anchor_rows"] += int(file_report["recoverable_board_anchor_rows"])
            source_counts["natural_text_rows"] += int(file_report["natural_text_rows"])
            source_counts["conversation_rows"] += int(file_report["conversation_rows"])
            source_counts["board_anchored_rows"] += int(file_report["board_anchored_rows"])
            source_counts["text_only_rows"] += int(file_report["text_only_rows"])
            source_counts["unusable_rows"] += int(file_report["unusable_rows"])
        overall_counts["board_anchored"] += source_counts["board_anchored_rows"]
        overall_counts["text_only"] += source_counts["text_only_rows"]
        overall_counts["unusable"] += source_counts["unusable_rows"]
        source_reports[source_name] = {
            "path": str(path),
            "available": True,
            "is_file": path.is_file(),
            "row_count": int(source_counts["row_count"]),
            "structured_file_count": len(source_file_reports),
            "files": source_file_reports,
            "explicit_fen_rows": int(source_counts["explicit_fen_rows"]),
            "explicit_move_list_rows": int(source_counts["explicit_move_list_rows"]),
            "recoverable_board_anchor_rows": int(source_counts["recoverable_board_anchor_rows"]),
            "natural_text_rows": int(source_counts["natural_text_rows"]),
            "conversation_rows": int(source_counts["conversation_rows"]),
            "board_anchored_rows": int(source_counts["board_anchored_rows"]),
            "text_only_rows": int(source_counts["text_only_rows"]),
            "unusable_rows": int(source_counts["unusable_rows"]),
        }

    return source_reports, board_anchored_rows, text_only_rows, overall_counts


def _build_audit_report_from_inventory(
    source_reports: Mapping[str, Any],
    overall_counts: Counter[str],
) -> dict[str, Any]:
    conclusions: list[str] = []
    waterhorse_report = source_reports.get("waterhorse_raw", {})
    if not waterhorse_report.get("available"):
        conclusions.append("The primary Waterhorse raw snapshot is not present locally; source expansion is blocked until it is fetched.")
    elif int(waterhorse_report.get("board_anchored_rows", 0)) <= 2:
        conclusions.append("Waterhorse is present, but board-anchored coverage is still too small to count as a material realism upgrade.")
    else:
        conclusions.append("Waterhorse annotated PGN contributes material board-anchored natural-text rows beyond the earlier sample-only coverage.")
    if int(overall_counts["board_anchored"]) <= 2:
        conclusions.append("Auxiliary board-anchored language rows remain sample-scale only.")
    return {
        "sources": source_reports,
        "overall_counts": dict(overall_counts),
        "conclusions": conclusions,
    }


def _build_audit_report(source_paths: Mapping[str, str | Path]) -> dict[str, Any]:
    source_reports, _board_rows, _text_rows, overall_counts = _collect_source_inventory(source_paths)
    return _build_audit_report_from_inventory(source_reports, overall_counts)


def audit_aux_language_sources(
    *,
    source_paths: Mapping[str, str | Path] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Audit local auxiliary language sources at file level."""
    resolved_paths = source_paths or DEFAULT_AUX_SOURCE_PATHS
    report_dir = Path(output_dir) if output_dir is not None else Path("data/pilot/language_probe_v4/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report = _build_audit_report(resolved_paths)
    json_path = report_dir / "aux_source_audit.json"
    md_path = report_dir / "aux_source_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_aux_source_markdown(report), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "report": report,
    }


def _aux_source_markdown(report: dict[str, Any]) -> str:
    lines = ["# Auxiliary Source Audit", ""]
    for source_name, source_report in report["sources"].items():
        lines.append(f"## {source_name}")
        lines.append(f"- available: {source_report['available']}")
        lines.append(f"- row_count: {source_report['row_count']}")
        lines.append(f"- structured_file_count: {source_report['structured_file_count']}")
        lines.append(f"- board_anchored_rows: {source_report['board_anchored_rows']}")
        lines.append(f"- text_only_rows: {source_report['text_only_rows']}")
        lines.append(f"- unusable_rows: {source_report['unusable_rows']}")
        for file_report in source_report.get("files", []):
            lines.append(
                f"  file: {Path(file_report['path']).name} / board_anchored={file_report['board_anchored_rows']} / "
                f"text_only={file_report['text_only_rows']} / unusable={file_report['unusable_rows']}"
            )
        lines.append("")
    lines.append("## Conclusions")
    for conclusion in report["conclusions"]:
        lines.append(f"- {conclusion}")
    return "\n".join(lines) + "\n"


def build_aux_language_corpora(
    *,
    source_paths: Mapping[str, str | Path] | None = None,
    output_root: str | Path = "data/pilot/language_probe_v4",
    config: AuxLanguageBuildConfig | None = None,
) -> dict[str, Any]:
    """Build board-anchored/text-only auxiliary language corpora from local sources."""
    resolved_paths = source_paths or DEFAULT_AUX_SOURCE_PATHS
    build_config = config or AuxLanguageBuildConfig()
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "reports"
    manifest_dir = output_dir / "manifests"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    source_reports, board_anchored_rows, text_only_rows, overall_counts = _collect_source_inventory(resolved_paths)
    audit_report = _build_audit_report_from_inventory(source_reports, overall_counts)
    split_rows = {"train": [], "val": [], "test": []}
    for row in board_anchored_rows:
        split_name = assign_split_by_game_id(str(row["split_group_id"]), build_config.split_config)
        split_rows[split_name].append({key: value for key, value in row.items() if key != "split_group_id"})

    outputs = {
        "aux_board_anchored_train": str(output_dir / "aux_board_anchored_train.jsonl"),
        "aux_board_anchored_val": str(output_dir / "aux_board_anchored_val.jsonl"),
        "aux_board_anchored_test": str(output_dir / "aux_board_anchored_test.jsonl"),
        "aux_text_only": str(output_dir / "aux_text_only.jsonl"),
    }
    write_jsonl(outputs["aux_board_anchored_train"], [{**row, "split": "train"} for row in split_rows["train"]])
    write_jsonl(outputs["aux_board_anchored_val"], [{**row, "split": "val"} for row in split_rows["val"]])
    write_jsonl(outputs["aux_board_anchored_test"], [{**row, "split": "test"} for row in split_rows["test"]])
    write_jsonl(outputs["aux_text_only"], text_only_rows)

    manifest = {
        "sources": {name: str(path) for name, path in resolved_paths.items()},
        "config": asdict(build_config.split_config),
        "source_reports": source_reports,
        "overall_counts": dict(overall_counts),
        "board_anchored_split_counts": {split_name: len(rows) for split_name, rows in split_rows.items()},
        "text_only_count": len(text_only_rows),
        "outputs": outputs,
    }
    manifest_path = manifest_dir / "aux_source_manifest.yaml"
    write_yaml(manifest_path, manifest)

    report_json = report_dir / "aux_source_audit.json"
    report_md = report_dir / "aux_source_audit.md"
    report_json.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
    report_md.write_text(_aux_source_markdown(audit_report), encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "outputs": outputs,
        "board_anchored_split_counts": manifest["board_anchored_split_counts"],
        "text_only_count": len(text_only_rows),
        "audit_report_path": str(report_json),
    }
