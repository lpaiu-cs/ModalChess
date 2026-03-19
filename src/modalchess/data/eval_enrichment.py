"""외부 평가 데이터를 ModalChess supervised JSONL에 조인한다."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from modalchess.data.preprocessing_common import (
    load_records_from_path,
    normalize_fen_for_eval_join,
    write_jsonl,
    write_yaml,
)


@dataclass(slots=True)
class EvalEnrichmentConfig:
    """Lichess evaluation enrichment 설정."""

    source_name: str = "lichess_eval"
    source_license: str = "CC0"
    source_version: str | None = None
    source_date: str | None = None
    mate_mode: str = "separate_field"
    mate_cp_sentinel: int = 100000
    keep_unmatched_rows: bool = True
    best_move_field: str = "engine_best_move_uci"


def _as_mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _extract_first_move_from_line(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        tokens = value.split()
        return tokens[0] if tokens else None
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                return text
    return None


def _depth_from_mapping(payload: Mapping[str, Any]) -> int:
    depth = payload.get("depth")
    try:
        return int(depth)
    except (TypeError, ValueError):
        return -1


def extract_eval_payload(row: Mapping[str, Any], config: EvalEnrichmentConfig) -> dict[str, Any]:
    """평가 row에서 조인 가능한 핵심 필드를 추출한다."""
    fen = str(row.get("fen") or row.get("FEN") or "").strip()
    if not fen:
        raise ValueError("eval row에 fen이 없다.")

    best_depth = _depth_from_mapping(row)
    container: Mapping[str, Any] = row
    eval_candidates = _as_mapping_list(row.get("evals"))
    if eval_candidates:
        container = max(eval_candidates, key=_depth_from_mapping)
        best_depth = _depth_from_mapping(container)

    leaf: Mapping[str, Any] = container
    pv_candidates = _as_mapping_list(container.get("pvs"))
    if pv_candidates:
        leaf = pv_candidates[0]

    cp_value = leaf.get("cp", container.get("cp", row.get("cp")))
    mate_value = leaf.get("mate", container.get("mate", row.get("mate")))
    best_move = (
        row.get(config.best_move_field)
        or container.get(config.best_move_field)
        or leaf.get(config.best_move_field)
        or _extract_first_move_from_line(leaf.get("line"))
        or _extract_first_move_from_line(leaf.get("pv"))
        or _extract_first_move_from_line(leaf.get("moves"))
        or _extract_first_move_from_line(container.get("pv"))
        or _extract_first_move_from_line(row.get("pv"))
    )

    return {
        "normalized_fen": normalize_fen_for_eval_join(fen),
        "depth": best_depth,
        "cp": int(cp_value) if cp_value is not None else None,
        "mate": int(mate_value) if mate_value is not None else None,
        "best_move": str(best_move) if best_move is not None else None,
    }


def build_eval_index(
    eval_paths: list[str | Path],
    config: EvalEnrichmentConfig | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """평가 파일들을 normalized FEN 키 인덱스로 정리한다."""
    enrich_config = config or EvalEnrichmentConfig()
    eval_index: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "source": enrich_config.source_name,
        "rows_seen": 0,
        "rows_indexed": 0,
        "drop_reasons": {},
    }

    def bump_drop(reason: str) -> None:
        report["drop_reasons"][reason] = int(report["drop_reasons"].get(reason, 0)) + 1

    for eval_path in eval_paths:
        for row in load_records_from_path(eval_path):
            report["rows_seen"] += 1
            try:
                payload = extract_eval_payload(row, enrich_config)
            except Exception as exc:
                bump_drop(type(exc).__name__)
                continue
            key = str(payload["normalized_fen"])
            current = eval_index.get(key)
            if current is None or int(payload["depth"]) > int(current["depth"]):
                eval_index[key] = payload
            report["rows_indexed"] = len(eval_index)
    return eval_index, report


def enrich_supervised_records_with_eval(
    supervised_paths: list[str | Path],
    eval_paths: list[str | Path],
    config: EvalEnrichmentConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """supervised JSONL에 평가 정보를 조인한다."""
    enrich_config = config or EvalEnrichmentConfig()
    eval_index, eval_report = build_eval_index(eval_paths, enrich_config)
    enriched_records: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "supervised_rows_seen": 0,
        "supervised_rows_written": 0,
        "matched_rows": 0,
        "unmatched_rows": 0,
        "mate_rows": 0,
        "eval_index": eval_report,
    }

    for supervised_path in supervised_paths:
        for row in load_records_from_path(supervised_path):
            report["supervised_rows_seen"] += 1
            enriched = dict(row)
            key = normalize_fen_for_eval_join(str(row["fen"]))
            payload = eval_index.get(key)
            if payload is None:
                report["unmatched_rows"] += 1
                if not enrich_config.keep_unmatched_rows:
                    continue
                enriched_records.append(enriched)
                report["supervised_rows_written"] += 1
                continue

            report["matched_rows"] += 1
            if payload.get("cp") is not None:
                enriched["engine_eval_cp"] = payload["cp"]
            elif payload.get("mate") is not None:
                report["mate_rows"] += 1
                if enrich_config.mate_mode == "sentinel":
                    mate_value = int(payload["mate"])
                    enriched["engine_eval_cp"] = (
                        enrich_config.mate_cp_sentinel if mate_value > 0 else -enrich_config.mate_cp_sentinel
                    )
                else:
                    enriched["engine_eval_mate"] = int(payload["mate"])
            if payload.get("best_move") is not None:
                enriched[enrich_config.best_move_field] = payload["best_move"]
            enriched_records.append(enriched)
            report["supervised_rows_written"] += 1
    return enriched_records, report


def write_enriched_supervised_jsonl(
    supervised_paths: list[str | Path],
    eval_paths: list[str | Path],
    output_path: str | Path,
    config: EvalEnrichmentConfig | None = None,
) -> dict[str, Any]:
    """조인된 enriched supervised JSONL과 manifest를 기록한다."""
    enrich_config = config or EvalEnrichmentConfig()
    enriched_records, report = enrich_supervised_records_with_eval(
        supervised_paths=supervised_paths,
        eval_paths=eval_paths,
        config=enrich_config,
    )
    output_file = Path(output_path)
    write_jsonl(output_file, enriched_records)
    manifest = {
        "source": {
            "name": enrich_config.source_name,
            "license": enrich_config.source_license,
            "version": enrich_config.source_version,
            "date": enrich_config.source_date,
            "supervised_inputs": [str(Path(path)) for path in supervised_paths],
            "eval_inputs": [str(Path(path)) for path in eval_paths],
        },
        "preprocessing": asdict(enrich_config),
        "outputs": {"jsonl": str(output_file)},
        "report": report,
    }
    write_yaml(output_file.parent / "eval_enrichment_manifest.yaml", manifest)
    return manifest
