"""Build a broader move-conditioned comment corpus from multiple audited sources."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable
import zipfile

import chess

from modalchess.data.annotated_source_fetch import default_annotated_source_specs
from modalchess.data.comment_source_audit import derive_comment_source_family
from modalchess.data.preprocessing_common import (
    StableSplitConfig,
    assign_split_by_game_id,
    iter_records_from_path,
    stable_hash_record,
    stable_hash_text,
    write_jsonl,
    write_yaml,
)


FEN_RE = re.compile(r'FEN of the given chess board is "([^"]+)"', re.IGNORECASE)
OUTPUT_RE = re.compile(r"Move([AB]):([a-h][1-8][a-h][1-8][nbrq]?)", re.IGNORECASE)
MOVE_BLOCK_RE = re.compile(
    r"MoveA:([a-h][1-8][a-h][1-8][nbrq]?)(.*)MoveB:([a-h][1-8][a-h][1-8][nbrq]?)(.*)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(slots=True)
class MultisourceAnnotatedSidecarConfig:
    split_config: StableSplitConfig = field(
        default_factory=lambda: StableSplitConfig(salt="modalchess_week15_multisource")
    )
    source_family_caps: dict[str, int] = field(
        default_factory=lambda: {"train": 30000, "val": 4000, "test": 4000}
    )
    min_source_family_presence: dict[str, int] = field(
        default_factory=lambda: {"train": 200, "val": 25, "test": 25}
    )


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    for split_name in ("train", "val", "test"):
        split_path = path / f"{split_name}.jsonl"
        if not split_path.exists():
            continue
        for row in iter_records_from_path(split_path):
            payload = dict(row)
            payload.setdefault("split", split_name)
            yield payload


def _waterhorse_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _iter_rows(path):
        payload = dict(row)
        payload["original_comment_text"] = str(row.get("comment_text") or "")
        payload["source_family"] = derive_comment_source_family(row)
        payload["record_semantics"] = "annotated_pgn_comment"
        rows.append(payload)
    return rows


def _clean_mate_segment(text: str, *, variant: str) -> str:
    value = text.strip().lstrip(",").strip()
    value = re.sub(r"\bTactic[AB]:", "Tactic:", value)
    if variant == "tactic" and value.lower().startswith("tactic "):
        value = "Tactic: " + value[7:].strip()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _extract_mate_row(row: dict[str, Any], *, source_name: str, source_family: str, source_file: str, line_index: int) -> dict[str, Any] | None:
    input_text = str(row.get("input") or "")
    output_text = str(row.get("output") or "")
    if not input_text or not output_text:
        return None
    fen_match = FEN_RE.search(input_text)
    block_match = MOVE_BLOCK_RE.search(input_text)
    output_match = OUTPUT_RE.search(output_text)
    if fen_match is None or block_match is None or output_match is None:
        return None

    fen = fen_match.group(1).strip()
    move_a = block_match.group(1).lower()
    text_a = block_match.group(2)
    move_b = block_match.group(3).lower()
    text_b = block_match.group(4)
    selected_label = output_match.group(1).upper()
    selected_move = output_match.group(2).lower()
    chosen_comment = text_a if selected_label == "A" else text_b
    chosen_comment = _clean_mate_segment(chosen_comment, variant=source_family)
    if not chosen_comment:
        return None

    candidate_moves = [move_a, move_b]
    if selected_move not in candidate_moves:
        return None

    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(selected_move)
    except ValueError:
        return None
    if move not in board.legal_moves:
        return None
    next_board = board.copy(stack=False)
    next_board.push(move)
    source_row_id = f"{Path(source_file).name}:{line_index}"
    game_id = stable_hash_text(f"{source_name}:{source_row_id}", prefix="mate_game_", length=16)
    position_id = stable_hash_text(f"{source_name}:{source_row_id}:0", prefix="pos_", length=16)
    sidecar_id = stable_hash_text(f"{source_name}:{source_row_id}:comment", prefix="sidecar_", length=16)
    return {
        "sidecar_id": sidecar_id,
        "probe_id": sidecar_id,
        "source": source_name,
        "source_family": source_family,
        "source_file": source_file,
        "source_row_id": source_row_id,
        "game_id": game_id,
        "position_id": position_id,
        "ply_index": 0,
        "fen": fen,
        "target_move_uci": selected_move,
        "next_fen": next_board.fen(en_passant="fen"),
        "comment_text": chosen_comment,
        "original_comment_text": chosen_comment,
        "comment_source": f"{source_family}_explanation",
        "candidate_moves": candidate_moves,
        "metadata": {
            "instruction": row.get("instruction"),
            "input": input_text,
            "output": output_text,
            "source_schema": "mate_pairwise_explanation",
        },
        "record_semantics": "pairwise_explanation",
    }


def _iter_mate_zip_rows(path: Path, *, source_name: str, source_family: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        jsonl_names = [
            name
            for name in archive.namelist()
            if name.endswith(".jsonl") and "__MACOSX" not in name and ".DS_Store" not in name
        ]
        for member_name in sorted(jsonl_names):
            variant = "both"
            lower_name = member_name.lower()
            if "strategy" in lower_name:
                variant = "strategy"
            elif "tactic" in lower_name:
                variant = "tactic"
            elif "noexplain" in lower_name:
                variant = "no_explain"
            if variant == "no_explain":
                continue
            with archive.open(member_name, "r") as handle:
                for line_index, raw_line in enumerate(handle):
                    if not raw_line.strip():
                        continue
                    payload = json.loads(raw_line.decode("utf-8"))
                    row = _extract_mate_row(
                        payload,
                        source_name=source_name,
                        source_family=source_family,
                        source_file=f"{path}::{member_name}",
                        line_index=line_index,
                    )
                    if row is not None:
                        rows.append(row)
    return rows


def _usable_source_specs() -> list[tuple[str, str, Path]]:
    usable: list[tuple[str, str, Path]] = []
    for spec in default_annotated_source_specs():
        if spec.usability != "usable":
            continue
        if not spec.local_path.exists():
            continue
        usable.append((spec.source_name, spec.source_family, spec.local_path))
    return usable


def _mate_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name, source_family, path in _usable_source_specs():
        if not source_name.startswith("mate_"):
            continue
        rows.extend(_iter_mate_zip_rows(path, source_name=source_name, source_family=source_family))
    return rows


def _ranked_row_hash(row: dict[str, Any], salt: str) -> str:
    return stable_hash_text(
        f"{salt}:{row.get('source_family')}:{row.get('sidecar_id') or row.get('probe_id')}",
        prefix="rank_",
        length=24,
    )


def _apply_source_family_caps(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    config: MultisourceAnnotatedSidecarConfig,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    capped_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    cap_report: dict[str, Any] = {}
    for split_name, rows in rows_by_split.items():
        family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            family_groups[str(row.get("source_family") or "unknown")].append(row)
        split_cap = int(config.source_family_caps.get(split_name, 0))
        min_presence = int(config.min_source_family_presence.get(split_name, 0))
        selected_rows: list[dict[str, Any]] = []
        family_counts_before: dict[str, int] = {}
        family_counts_after: dict[str, int] = {}
        for family_name, group_rows in sorted(family_groups.items()):
            family_counts_before[family_name] = len(group_rows)
            ranked_rows = sorted(group_rows, key=lambda row: _ranked_row_hash(row, config.split_config.salt))
            if split_cap > 0:
                target = min(len(ranked_rows), max(min_presence, split_cap))
            else:
                target = len(ranked_rows)
            chosen_rows = ranked_rows[:target]
            family_counts_after[family_name] = len(chosen_rows)
            selected_rows.extend(chosen_rows)
        selected_rows.sort(key=lambda row: _ranked_row_hash(row, config.split_config.salt))
        capped_rows_by_split[split_name] = selected_rows
        cap_report[split_name] = {
            "family_counts_before": family_counts_before,
            "family_counts_after": family_counts_after,
        }
    return capped_rows_by_split, cap_report


def _split_assignment_for_mate(rows: list[dict[str, Any]], split_config: StableSplitConfig) -> dict[str, list[dict[str, Any]]]:
    rows_by_split = {"train": [], "val": [], "test": []}
    for row in rows:
        split_name = assign_split_by_game_id(str(row["game_id"]), split_config)
        payload = dict(row)
        payload["split"] = split_name
        rows_by_split[split_name].append(payload)
    return rows_by_split


def _count_by(rows: Iterable[dict[str, Any]], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter(str(row.get(field_name) or "unknown") for row in rows)
    return dict(counter.most_common())


def build_multisource_annotated_sidecar(
    *,
    waterhorse_input_root: str | Path = "data/pilot/annotated_sidecar_v1",
    output_root: str | Path = "data/pilot/annotated_sidecar_v4_multisource",
    config: MultisourceAnnotatedSidecarConfig | None = None,
) -> dict[str, Any]:
    multisource_config = config or MultisourceAnnotatedSidecarConfig()
    output_dir = Path(output_root)
    reports_dir = output_dir / "reports"
    manifests_dir = output_dir / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    waterhorse_rows = _waterhorse_rows(Path(waterhorse_input_root))
    mate_rows = _mate_rows()

    rows_by_split = {"train": [], "val": [], "test": []}
    for row in waterhorse_rows:
        rows_by_split[str(row["split"])].append(row)
    mate_split_rows = _split_assignment_for_mate(mate_rows, multisource_config.split_config)
    for split_name in ("train", "val", "test"):
        rows_by_split[split_name].extend(mate_split_rows[split_name])

    capped_rows_by_split, cap_report = _apply_source_family_caps(rows_by_split, config=multisource_config)
    split_counts = {
        split_name: write_jsonl(output_dir / f"{split_name}.jsonl", capped_rows_by_split[split_name])
        for split_name in ("train", "val", "test")
    }

    all_rows = [row for rows in capped_rows_by_split.values() for row in rows]
    rows_by_source_family = _count_by(all_rows, "source_family")
    rows_by_source = _count_by(all_rows, "source")
    rows_by_comment_source = _count_by(all_rows, "comment_source")
    family_count = len(rows_by_source_family)
    largest_family_share = (
        max(rows_by_source_family.values()) / sum(rows_by_source_family.values())
        if rows_by_source_family
        else 0.0
    )
    report = {
        "waterhorse_input_root": str(waterhorse_input_root),
        "output_root": str(output_dir),
        "config": {
            "split_config": asdict(multisource_config.split_config),
            "source_family_caps": multisource_config.source_family_caps,
            "min_source_family_presence": multisource_config.min_source_family_presence,
        },
        "split_counts": split_counts,
        "rows_by_source": rows_by_source,
        "rows_by_source_family": rows_by_source_family,
        "rows_by_comment_source": rows_by_comment_source,
        "source_family_diversity": {
            "family_count": family_count,
            "largest_family_share": largest_family_share,
            "single_family_dominant": largest_family_share >= 0.5,
        },
        "cap_report": cap_report,
        "source_semantics": {
            "annotated_pgn_comment": sum(int(row["record_semantics"] == "annotated_pgn_comment") for row in all_rows),
            "pairwise_explanation": sum(int(row["record_semantics"] == "pairwise_explanation") for row in all_rows),
        },
    }

    prior_rows = list(_iter_rows(Path("data/pilot/annotated_sidecar_v3_informative")))
    prior_families = Counter(str(row.get("source_family") or derive_comment_source_family(row)) for row in prior_rows)
    diff = {
        "v3_total_rows": len(prior_rows),
        "v4_total_rows": len(all_rows),
        "v3_family_count": len(prior_families),
        "v4_family_count": family_count,
        "v3_largest_family_share": (
            max(prior_families.values()) / sum(prior_families.values()) if prior_families else 0.0
        ),
        "v4_largest_family_share": largest_family_share,
    }

    manifest = {
        "waterhorse_input_root": str(waterhorse_input_root),
        "config": {
            "split_config": asdict(multisource_config.split_config),
            "source_family_caps": multisource_config.source_family_caps,
            "min_source_family_presence": multisource_config.min_source_family_presence,
        },
        "split_counts": split_counts,
        "outputs": {
            split_name: str(output_dir / f"{split_name}.jsonl")
            for split_name in ("train", "val", "test")
        },
        "rows_by_source_family": rows_by_source_family,
        "rows_by_source": rows_by_source,
    }
    manifest_path = manifests_dir / "multisource_sidecar_manifest.yaml"
    report_json_path = reports_dir / "multisource_sidecar_report.json"
    report_md_path = reports_dir / "multisource_sidecar_report.md"
    diff_json_path = reports_dir / "v3_informative_vs_v4_multisource_diff.json"
    write_yaml(manifest_path, manifest)
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    diff_json_path.write_text(json.dumps(diff, indent=2), encoding="utf-8")

    lines = ["# Multisource Annotated Sidecar Report", ""]
    lines.append(f"- split_counts: {split_counts}")
    lines.append(f"- rows_by_source_family: {rows_by_source_family}")
    lines.append(f"- rows_by_source: {rows_by_source}")
    lines.append(
        f"- source_family_diversity: family_count={family_count}, largest_family_share={largest_family_share:.4f}"
    )
    lines.append(f"- source_semantics: {report['source_semantics']}")
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "manifest_path": str(manifest_path),
        "report_json": str(report_json_path),
        "report_md": str(report_md_path),
        "diff_json": str(diff_json_path),
        "split_counts": split_counts,
        "rows_by_source_family": rows_by_source_family,
    }
