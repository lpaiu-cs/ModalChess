"""Build and report move-conditioned comment sidecars from annotated PGN."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import io
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

import chess
import chess.pgn
import yaml

from modalchess.data.preprocessing_common import (
    StableSplitConfig,
    assign_split_by_game_id,
    iter_records_from_path,
    special_rule_flags,
    stable_hash_record,
    stable_hash_text,
    write_jsonl,
    write_yaml,
)


DEFAULT_ANNOTATED_PGN_ROOT = Path("data/pilot/raw/hf/waterhorse_chess_data/chessgpt_data/annotated_pgn")


@dataclass(slots=True)
class AnnotatedPgnSidecarConfig:
    """Configuration for annotated PGN sidecar builds."""

    split_config: StableSplitConfig = field(
        default_factory=lambda: StableSplitConfig(salt="modalchess_week9_annotated_sidecar")
    )
    include_history_fens: bool = True


def _iter_annotated_source_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    files = [
        file_path
        for file_path in root.rglob("*")
        if file_path.is_file() and (file_path.name.endswith(".jsonl") or ".jsonl-" in file_path.name)
    ]
    return sorted(files)


def _iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    if ".jsonl-" in path.name:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    yield from iter_records_from_path(path)


def _normalize_comment_text(text: str | None) -> str:
    if text is None:
        return ""
    return text.strip()


def _node_comment_payload(node: chess.pgn.ChildNode) -> tuple[str, list[int], str | None]:
    comment_parts: list[str] = []
    source_parts: list[str] = []
    starting_comment = _normalize_comment_text(getattr(node, "starting_comment", ""))
    trailing_comment = _normalize_comment_text(getattr(node, "comment", ""))
    if starting_comment:
        comment_parts.append(starting_comment)
        source_parts.append("starting_comment")
    if trailing_comment:
        comment_parts.append(trailing_comment)
        source_parts.append("comment")
    nag_codes = sorted(int(nag_code) for nag_code in getattr(node, "nags", set()))
    if nag_codes:
        source_parts.append("nag")
    comment_text = "\n\n".join(comment_parts)
    comment_source = "+".join(source_parts) if source_parts else None
    return comment_text, nag_codes, comment_source


def _is_standard_game(game: chess.pgn.Game) -> bool:
    variant = str(game.headers.get("Variant") or "").strip()
    return variant in {"", "Standard"}


def _game_id(source_file: Path, row: Mapping[str, Any], game: chess.pgn.Game) -> str:
    headers = {
        "source_file": source_file.name,
        "pipeline_key": row.get("pipeline_key"),
        "event": game.headers.get("Event"),
        "site": game.headers.get("Site"),
        "round": game.headers.get("Round"),
        "white": game.headers.get("White"),
        "black": game.headers.get("Black"),
        "date": game.headers.get("Date"),
        "fen": game.headers.get("FEN"),
    }
    return stable_hash_record(headers, prefix="ann_game_", length=16)


def _normalized_fen(board: chess.Board) -> str:
    normalized_board = board.copy(stack=False)
    normalized_board.castling_rights = normalized_board.clean_castling_rights()
    return normalized_board.fen(en_passant="fen")


def _position_id(game_id: str, ply_index: int) -> str:
    return stable_hash_text(f"{game_id}:{ply_index}", prefix="pos_", length=16)


def _sidecar_id(game_id: str, ply_index: int) -> str:
    return stable_hash_text(f"{game_id}:{ply_index}:comment", prefix="sidecar_", length=16)


def _comment_length_stats(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    sorted_lengths = sorted(lengths)

    def _percentile(percentile: float) -> float:
        if len(sorted_lengths) == 1:
            return float(sorted_lengths[0])
        rank = int(round((len(sorted_lengths) - 1) * percentile))
        return float(sorted_lengths[rank])

    return {
        "mean": float(statistics.fmean(sorted_lengths)),
        "p50": _percentile(0.50),
        "p90": _percentile(0.90),
        "p95": _percentile(0.95),
        "max": float(sorted_lengths[-1]),
    }


def build_annotated_pgn_sidecar(
    *,
    input_root: str | Path = DEFAULT_ANNOTATED_PGN_ROOT,
    output_root: str | Path = "data/pilot/annotated_sidecar_v1",
    config: AnnotatedPgnSidecarConfig | None = None,
) -> dict[str, Any]:
    """Build a move-conditioned comment sidecar from annotated PGN rows."""
    sidecar_config = config or AnnotatedPgnSidecarConfig()
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "manifests"
    report_dir = output_dir / "reports"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    split_rows = {"train": [], "val": [], "test": []}
    drop_counts: Counter[str] = Counter()
    comment_source_counts: Counter[str] = Counter()
    game_comment_counts: Counter[str] = Counter()
    total_source_rows = 0
    parsed_games = 0
    emitted_games: set[str] = set()

    for source_file in _iter_annotated_source_files(input_root):
        for row_index, row in enumerate(_iter_json_records(source_file)):
            total_source_rows += 1
            source_row_id = str(row.get("pipeline_key") or stable_hash_record(row, prefix="row_", length=16))
            pgn_text = str(row.get("text") or "")
            if not pgn_text.strip():
                drop_counts["missing_text"] += 1
                continue
            try:
                game = chess.pgn.read_game(io.StringIO(pgn_text))
            except Exception:
                drop_counts["parse_error"] += 1
                continue
            if game is None:
                drop_counts["parse_error"] += 1
                continue
            if getattr(game, "errors", []):
                drop_counts["pgn_game_errors"] += 1
                continue
            if not _is_standard_game(game):
                drop_counts["non_standard_variant"] += 1
                continue

            parsed_games += 1
            game_id = _game_id(source_file, row, game)
            split_name = assign_split_by_game_id(game_id, sidecar_config.split_config)
            board = game.board()
            if board.chess960:
                drop_counts["chess960_or_ambiguous_setup"] += 1
                continue
            history_fens = [_normalized_fen(board)]
            emitted_for_game = 0
            node: chess.pgn.GameNode = game
            ply_index = 0

            while node.variations:
                child = node.variation(0)
                move = child.move
                ply_index += 1
                current_fen = history_fens[-1]
                comment_text, nag_codes, comment_source = _node_comment_payload(child)
                if comment_source:
                    if move not in board.legal_moves:
                        drop_counts["illegal_move"] += 1
                        break
                    next_board = board.copy(stack=False)
                    next_board.push(move)
                    next_fen = _normalized_fen(next_board)
                    position_id = _position_id(game_id, ply_index)
                    sidecar_id = _sidecar_id(game_id, ply_index)
                    record: dict[str, Any] = {
                        "sidecar_id": sidecar_id,
                        "probe_id": sidecar_id,
                        "source": "waterhorse_annotated_pgn",
                        "source_file": str(source_file),
                        "source_row_id": source_row_id,
                        "game_id": game_id,
                        "position_id": position_id,
                        "ply_index": ply_index,
                        "fen": current_fen,
                        "target_move_uci": move.uci(),
                        "next_fen": next_fen,
                        "comment_text": comment_text,
                        "nag_codes": nag_codes,
                        "history_fens": list(history_fens) if sidecar_config.include_history_fens else None,
                        "split": split_name,
                        "comment_source": comment_source,
                        "metadata": {
                            "pipeline_key": row.get("pipeline_key"),
                            "source_record_index": row_index,
                            "headers": dict(game.headers),
                            "annotator": game.headers.get("Annotator"),
                            "variant": game.headers.get("Variant"),
                            "result": game.headers.get("Result"),
                        },
                    }
                    if sidecar_config.include_history_fens and record["history_fens"][-1] != current_fen:
                        drop_counts["history_tail_mismatch"] += 1
                        break
                    split_rows[split_name].append(record)
                    comment_source_counts[comment_source] += 1
                    game_comment_counts[game_id] += 1
                    emitted_for_game += 1

                board.push(move)
                history_fens.append(_normalized_fen(board))
                node = child

            if emitted_for_game > 0:
                emitted_games.add(game_id)
            else:
                drop_counts["no_comment_annotations"] += 1

    outputs = {
        "train": str(output_dir / "train.jsonl"),
        "val": str(output_dir / "val.jsonl"),
        "test": str(output_dir / "test.jsonl"),
    }
    split_counts = {
        split_name: write_jsonl(outputs[split_name], split_rows[split_name])
        for split_name in ("train", "val", "test")
    }
    manifest = {
        "input_root": str(input_root),
        "config": {
            "split_config": asdict(sidecar_config.split_config),
            "include_history_fens": sidecar_config.include_history_fens,
        },
        "source_files": sorted(str(path) for path in _iter_annotated_source_files(input_root)),
        "total_source_rows": total_source_rows,
        "parsed_games": parsed_games,
        "emitted_game_count": len(emitted_games),
        "split_counts": split_counts,
        "drop_counts": dict(drop_counts),
        "comment_source_counts": dict(comment_source_counts),
        "avg_comment_density_per_commented_game": (
            float(statistics.fmean(game_comment_counts.values())) if game_comment_counts else 0.0
        ),
        "outputs": outputs,
    }
    manifest_path = manifest_dir / "annotated_sidecar_manifest.yaml"
    write_yaml(manifest_path, manifest)
    return {
        "manifest_path": str(manifest_path),
        "outputs": outputs,
        "split_counts": split_counts,
        "drop_counts": dict(drop_counts),
    }


def _load_sidecar_rows(root: str | Path) -> dict[str, list[dict[str, Any]]]:
    sidecar_root = Path(root)
    return {
        split_name: [dict(row) for row in iter_records_from_path(sidecar_root / f"{split_name}.jsonl")]
        for split_name in ("train", "val", "test")
    }


def _load_aux_rows(root: str | Path) -> list[dict[str, Any]]:
    aux_root = Path(root)
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        path = aux_root / f"aux_board_anchored_{split_name}.jsonl"
        if not path.exists():
            continue
        rows.extend(dict(row) for row in iter_records_from_path(path))
    return rows


def _load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.loads(json.dumps(yaml.safe_load(handle) or {}))
    return payload if isinstance(payload, dict) else {}


def generate_annotated_sidecar_report(
    *,
    input_root: str | Path,
    compare_aux_root: str | Path | None = None,
) -> dict[str, Any]:
    """Generate QA report for the annotated PGN sidecar."""
    sidecar_root = Path(input_root)
    rows_by_split = _load_sidecar_rows(sidecar_root)
    manifest = _load_manifest(sidecar_root / "manifests" / "annotated_sidecar_manifest.yaml")

    total_rows_by_split = {split_name: len(rows) for split_name, rows in rows_by_split.items()}
    unique_game_ids_by_split = {
        split_name: len({str(row["game_id"]) for row in rows})
        for split_name, rows in rows_by_split.items()
    }

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    comment_char_lengths = [len(str(row.get("comment_text") or "")) for row in all_rows]
    comment_token_lengths = [len(str(row.get("comment_text") or "").split()) for row in all_rows]
    valid_target_move_rows = 0
    valid_next_fen_rows = 0
    duplicate_counter: Counter[tuple[str, str, str]] = Counter()
    comment_density_counter: Counter[str] = Counter()
    special_rule_counts_by_split: dict[str, dict[str, int]] = {
        split_name: {"promotion": 0, "castling": 0, "en_passant": 0, "check_evasion": 0}
        for split_name in ("train", "val", "test")
    }

    for split_name, rows in rows_by_split.items():
        for row in rows:
            board = chess.Board(str(row["fen"]))
            move = chess.Move.from_uci(str(row["target_move_uci"]))
            valid_target_move_rows += int(move in board.legal_moves)
            next_board = board.copy(stack=False)
            next_board.push(move)
            valid_next_fen_rows += int(next_board.fen(en_passant="fen") == str(row["next_fen"]))
            flags = special_rule_flags(str(row["fen"]), str(row["target_move_uci"]))
            for key, value in flags.items():
                special_rule_counts_by_split[split_name][key] += int(value)
            duplicate_key = (
                str(row["fen"]),
                str(row["target_move_uci"]),
                str(row.get("comment_text") or "").strip(),
            )
            duplicate_counter[duplicate_key] += 1
            comment_density_counter[str(row["game_id"])] += 1

    duplicate_rows = sum(count - 1 for count in duplicate_counter.values() if count > 1)
    duplicate_rate = (duplicate_rows / len(all_rows)) if all_rows else 0.0
    density_values = list(comment_density_counter.values())
    comment_density_stats = _comment_length_stats(density_values)

    aux_comparison = None
    if compare_aux_root is not None:
        aux_rows = _load_aux_rows(compare_aux_root)
        aux_move_conditioned_rows = sum(
            int(bool(row.get("target_move_uci")) and bool(str(row.get("text") or "").strip()))
            for row in aux_rows
        )
        annotated_move_conditioned_rows = sum(
            int(bool(row.get("target_move_uci")) and bool(str(row.get("comment_text") or "").strip()))
            for row in all_rows
        )
        aux_comparison = {
            "week8_aux_root": str(compare_aux_root),
            "week8_aux_board_anchored_rows": len(aux_rows),
            "week8_move_conditioned_text_rows": aux_move_conditioned_rows,
            "annotated_sidecar_rows": len(all_rows),
            "annotated_move_conditioned_text_rows": annotated_move_conditioned_rows,
            "move_conditioned_text_row_delta": annotated_move_conditioned_rows - aux_move_conditioned_rows,
            "material_move_conditioned_coverage_increase": annotated_move_conditioned_rows > aux_move_conditioned_rows,
        }

    report = {
        "input_root": str(sidecar_root),
        "total_rows_by_split": total_rows_by_split,
        "unique_game_ids_by_split": unique_game_ids_by_split,
        "comment_length_chars": _comment_length_stats(comment_char_lengths),
        "comment_length_tokens": _comment_length_stats(comment_token_lengths),
        "rows_with_valid_target_move_uci": valid_target_move_rows,
        "rows_with_valid_next_fen": valid_next_fen_rows,
        "drop_counts": manifest.get("drop_counts", {}),
        "special_rule_coverage_by_split": special_rule_counts_by_split,
        "duplicate_position_comment_rate": duplicate_rate,
        "duplicate_position_comment_rows": duplicate_rows,
        "comment_density_per_game": comment_density_stats,
        "comment_source_counts": manifest.get("comment_source_counts", {}),
        "week8_aux_comparison": aux_comparison,
    }
    return report


def _annotated_sidecar_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Annotated Sidecar Report", ""]
    lines.append("## Counts")
    for split_name, count in report["total_rows_by_split"].items():
        lines.append(
            f"- `{split_name}`: rows={count}, unique_games={report['unique_game_ids_by_split'][split_name]}"
        )
    lines.append("")
    lines.append("## Comment Length")
    lines.append(
        f"- chars: mean={report['comment_length_chars']['mean']:.2f}, "
        f"p50={report['comment_length_chars']['p50']:.2f}, "
        f"p90={report['comment_length_chars']['p90']:.2f}, "
        f"p95={report['comment_length_chars']['p95']:.2f}, "
        f"max={report['comment_length_chars']['max']:.2f}"
    )
    lines.append(
        f"- tokens: mean={report['comment_length_tokens']['mean']:.2f}, "
        f"p50={report['comment_length_tokens']['p50']:.2f}, "
        f"p90={report['comment_length_tokens']['p90']:.2f}, "
        f"p95={report['comment_length_tokens']['p95']:.2f}, "
        f"max={report['comment_length_tokens']['max']:.2f}"
    )
    lines.append("")
    lines.append("## Validation")
    lines.append(f"- valid_target_move_rows: {report['rows_with_valid_target_move_uci']}")
    lines.append(f"- valid_next_fen_rows: {report['rows_with_valid_next_fen']}")
    lines.append(f"- duplicate_position_comment_rate: {report['duplicate_position_comment_rate']:.6f}")
    lines.append("")
    lines.append("## Drop Counts")
    for reason, count in sorted(report["drop_counts"].items()):
        lines.append(f"- `{reason}`: {count}")
    lines.append("")
    lines.append("## Special Rules")
    for split_name, counts in report["special_rule_coverage_by_split"].items():
        lines.append(
            f"- `{split_name}`: promotion={counts['promotion']}, castling={counts['castling']}, "
            f"en_passant={counts['en_passant']}, check_evasion={counts['check_evasion']}"
        )
    if report.get("week8_aux_comparison"):
        comparison = report["week8_aux_comparison"]
        lines.append("")
        lines.append("## Week-8 Comparison")
        lines.append(f"- week8_move_conditioned_text_rows: {comparison['week8_move_conditioned_text_rows']}")
        lines.append(f"- annotated_move_conditioned_text_rows: {comparison['annotated_move_conditioned_text_rows']}")
        lines.append(f"- move_conditioned_text_row_delta: {comparison['move_conditioned_text_row_delta']}")
        lines.append(
            f"- material_move_conditioned_coverage_increase: {comparison['material_move_conditioned_coverage_increase']}"
        )
    return "\n".join(lines) + "\n"


def write_annotated_sidecar_report(
    *,
    input_root: str | Path,
    output_dir: str | Path | None = None,
    compare_aux_root: str | Path | None = None,
) -> dict[str, str]:
    """Write JSON/Markdown reports for the annotated sidecar."""
    sidecar_root = Path(input_root)
    report_root = Path(output_dir) if output_dir is not None else sidecar_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    report = generate_annotated_sidecar_report(
        input_root=sidecar_root,
        compare_aux_root=compare_aux_root,
    )
    report_json = report_root / "annotated_sidecar_report.json"
    report_md = report_root / "annotated_sidecar_report.md"
    diff_json = report_root / "week8_vs_annotated_sidecar_diff.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md.write_text(_annotated_sidecar_markdown(report), encoding="utf-8")
    diff_payload = report.get("week8_aux_comparison") or {}
    diff_json.write_text(json.dumps(diff_payload, indent=2), encoding="utf-8")
    return {
        "report_json": str(report_json),
        "report_md": str(report_md),
        "diff_json": str(diff_json),
    }
