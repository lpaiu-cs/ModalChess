"""Audit and build auxiliary language corpora for week-7."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
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


DEFAULT_AUX_SOURCE_PATHS = {
    "waterhorse_raw": Path("data/pilot/raw/hf/waterhorse_chess_data"),
    "chessgpt_text_sample": Path("data/pilot/samples/chessgpt_text_corpus.jsonl"),
    "chessgpt_conversation_sample": Path("data/pilot/samples/chessgpt_conversation_corpus.jsonl"),
}


@dataclass(slots=True)
class AuxLanguageBuildConfig:
    """Auxiliary language corpus build configuration."""

    split_config: StableSplitConfig = field(
        default_factory=lambda: StableSplitConfig(salt="modalchess_week7_aux")
    )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.is_dir():
        return []
    return [dict(row) for row in iter_records_from_path(path)]


def _flatten_text_row(row: Mapping[str, Any]) -> str | None:
    segments = [str(row.get("prompt") or "").strip(), str(row.get("response") or "").strip(), str(row.get("text") or "").strip()]
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


def _classify_row(row: Mapping[str, Any]) -> str:
    text, _messages = _natural_text_and_messages(row)
    has_text = bool(text)
    has_fen = row.get("fen") not in (None, "")
    if has_fen and has_text:
        return "board_anchored"
    if has_text:
        return "text_only"
    return "unusable"


def audit_aux_language_sources(
    *,
    source_paths: Mapping[str, str | Path] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Audit local auxiliary language sources without changing semantics."""
    resolved_paths = {name: Path(path) for name, path in (source_paths or DEFAULT_AUX_SOURCE_PATHS).items()}
    report_dir = Path(output_dir) if output_dir is not None else Path("data/pilot/language_probe_v3/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    source_reports: dict[str, Any] = {}
    overall_counts = Counter()
    for source_name, path in resolved_paths.items():
        available = path.exists()
        rows = _load_rows(path) if available and path.is_file() else []
        category_counts = Counter()
        sample_fields: list[str] = sorted(rows[0].keys()) if rows else []
        natural_text_rows = 0
        explicit_fen_rows = 0
        explicit_move_list_rows = 0
        conversation_rows = 0
        for row in rows:
            category = _classify_row(row)
            category_counts[category] += 1
            text, messages = _natural_text_and_messages(row)
            natural_text_rows += int(bool(text))
            explicit_fen_rows += int(row.get("fen") not in (None, ""))
            explicit_move_list_rows += int(bool(parse_space_or_comma_separated(row.get("candidate_moves"))))
            conversation_rows += int(messages is not None)
        overall_counts.update(category_counts)
        source_reports[source_name] = {
            "path": str(path),
            "available": available,
            "is_file": path.is_file(),
            "row_count": len(rows),
            "schema_hint": str(rows[0].get("schema")) if rows else None,
            "sample_fields": sample_fields,
            "explicit_fen_rows": explicit_fen_rows,
            "explicit_move_list_rows": explicit_move_list_rows,
            "recoverable_board_anchor_rows": explicit_fen_rows,
            "natural_text_rows": natural_text_rows,
            "conversation_rows": conversation_rows,
            "board_anchored_rows": category_counts["board_anchored"],
            "text_only_rows": category_counts["text_only"],
            "unusable_rows": category_counts["unusable"],
        }

    conclusions: list[str] = []
    if not resolved_paths["waterhorse_raw"].exists():
        conclusions.append("The primary Waterhorse raw snapshot is not present locally; week-7 can only audit previously normalized local ChessGPT-style sample corpora.")
    if overall_counts["board_anchored"] == 0:
        conclusions.append("No auxiliary board-anchored language rows were available beyond existing MATE/puzzle sources.")
    else:
        conclusions.append("Local auxiliary corpora contain some board-anchored rows, but coverage is tiny and should be treated as sample-only.")

    report = {
        "sources": source_reports,
        "overall_counts": dict(overall_counts),
        "conclusions": conclusions,
    }
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
        lines.append(f"- board_anchored_rows: {source_report['board_anchored_rows']}")
        lines.append(f"- text_only_rows: {source_report['text_only_rows']}")
        lines.append(f"- unusable_rows: {source_report['unusable_rows']}")
        lines.append(f"- explicit_fen_rows: {source_report['explicit_fen_rows']}")
        lines.append(f"- natural_text_rows: {source_report['natural_text_rows']}")
        lines.append(f"- conversation_rows: {source_report['conversation_rows']}")
        lines.append("")
    lines.append("## Conclusions")
    for conclusion in report["conclusions"]:
        lines.append(f"- {conclusion}")
    return "\n".join(lines) + "\n"


def build_aux_language_corpora(
    *,
    source_paths: Mapping[str, str | Path] | None = None,
    output_root: str | Path = "data/pilot/language_probe_v3",
    config: AuxLanguageBuildConfig | None = None,
) -> dict[str, Any]:
    """Build board-anchored/text-only auxiliary language corpora from auditable local sources."""
    resolved_paths = {name: Path(path) for name, path in (source_paths or DEFAULT_AUX_SOURCE_PATHS).items()}
    build_config = config or AuxLanguageBuildConfig()
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)

    board_anchored_rows: list[dict[str, Any]] = []
    text_only_rows: list[dict[str, Any]] = []
    unusable_counts: Counter[str] = Counter()
    source_counts = Counter()

    for source_name, path in resolved_paths.items():
        if not path.exists() or not path.is_file():
            continue
        for row in _load_rows(path):
            category = _classify_row(row)
            if category == "unusable":
                unusable_counts[source_name] += 1
                continue
            text, messages = _natural_text_and_messages(row)
            source_row_id = str(row.get("position_id") or stable_hash_text(json.dumps(dict(row), sort_keys=True)))
            payload = {
                "aux_id": stable_hash_text(f"{source_name}:{source_row_id}", prefix="aux_", length=16),
                "source": str(row.get("source") or source_name),
                "source_file": str(path),
                "source_schema": str(row.get("schema") or "unknown"),
                "source_row_id": source_row_id,
                "fen": row.get("fen"),
                "fen_4field": normalize_fen_for_eval_join(str(row["fen"])) if row.get("fen") else None,
                "candidate_moves": parse_space_or_comma_separated(row.get("candidate_moves")),
                "text": text,
                "messages": messages,
                "board_anchor_type": "fen" if row.get("fen") else None,
                "text_kind": "conversation" if messages is not None else "natural_text",
                "metadata": {
                    "prompt": row.get("prompt"),
                    "response": row.get("response"),
                    "schema": row.get("schema"),
                },
            }
            source_counts[source_name] += 1
            if category == "text_only":
                text_only_rows.append(payload)
                continue
            split_name = assign_split_by_game_id(source_row_id, build_config.split_config)
            board_anchored_rows.append(
                {
                    **payload,
                    "split": split_name,
                }
            )

    split_rows = {"train": [], "val": [], "test": []}
    for row in board_anchored_rows:
        split_rows[str(row["split"])].append(row)

    outputs = {
        "aux_board_anchored_train": str(output_dir / "aux_board_anchored_train.jsonl"),
        "aux_board_anchored_val": str(output_dir / "aux_board_anchored_val.jsonl"),
        "aux_board_anchored_test": str(output_dir / "aux_board_anchored_test.jsonl"),
        "aux_text_only": str(output_dir / "aux_text_only.jsonl"),
    }
    write_jsonl(outputs["aux_board_anchored_train"], split_rows["train"])
    write_jsonl(outputs["aux_board_anchored_val"], split_rows["val"])
    write_jsonl(outputs["aux_board_anchored_test"], split_rows["test"])
    write_jsonl(outputs["aux_text_only"], text_only_rows)

    manifest = {
        "sources": {name: str(path) for name, path in resolved_paths.items()},
        "config": asdict(build_config.split_config),
        "source_counts": dict(source_counts),
        "unusable_counts": dict(unusable_counts),
        "board_anchored_split_counts": {split_name: len(rows) for split_name, rows in split_rows.items()},
        "text_only_count": len(text_only_rows),
        "outputs": outputs,
    }
    manifest_path = output_dir / "manifests" / "aux_source_manifest.yaml"
    write_yaml(manifest_path, manifest)
    return {
        "manifest_path": str(manifest_path),
        "outputs": outputs,
        "board_anchored_split_counts": manifest["board_anchored_split_counts"],
        "text_only_count": len(text_only_rows),
    }
