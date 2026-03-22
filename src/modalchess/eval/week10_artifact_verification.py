"""Verification helpers for week-10 annotated-sidecar artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml

from modalchess.data.annotated_pgn_sidecar import (
    generate_annotated_sidecar_report,
    write_annotated_sidecar_report,
)
from modalchess.data.comment_retrieval_eval import (
    CommentRetrievalEvalConfig,
    build_comment_retrieval_eval_regime,
)
from modalchess.eval.raw_text_retrieval import run_raw_text_retrieval_probes


REQUIRED_RELATIVE_PATHS = {
    "annotated_report_json": Path("data/pilot/annotated_sidecar_v1/reports/annotated_sidecar_report.json"),
    "annotated_report_md": Path("data/pilot/annotated_sidecar_v1/reports/annotated_sidecar_report.md"),
    "retrieval_manifest": Path("outputs/week10/comment_retrieval/retrieval_eval_manifest.yaml"),
    "retrieval_results_json": Path("outputs/week10/comment_retrieval/comment_retrieval_results.json"),
    "retrieval_results_csv": Path("outputs/week10/comment_retrieval/comment_retrieval_results.csv"),
    "retrieval_summary_md": Path("outputs/week10/comment_retrieval/comment_retrieval_summary.md"),
    "reference_summary_md": Path("outputs/week10/reference_from_week9/reference_summary.md"),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return json.loads(json.dumps(payload))


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _jsonl_row_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def choose_verification_gate(
    *,
    missing_count: int,
    mismatch_count: int,
    suspicious_findings: list[str],
) -> str:
    if missing_count > 0 or mismatch_count > 0:
        return "NOT_VERIFIED"
    if suspicious_findings:
        return "VERIFIED_BUT_NEEDS_SANITY_AUDIT"
    return "VERIFIED_GOOD"


def verify_week10_artifacts(
    *,
    repo_root: str | Path = ".",
    output_dir: str | Path = "outputs/week11/verification",
    regenerate_if_missing: bool = True,
    regenerate_if_stale: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    paths = {name: root / relative_path for name, relative_path in REQUIRED_RELATIVE_PATHS.items()}
    existence = {name: path.exists() for name, path in paths.items()}
    regenerated: list[str] = []
    missing = [name for name, exists in existence.items() if not exists]

    if regenerate_if_missing:
        if not existence["annotated_report_json"] or not existence["annotated_report_md"]:
            write_annotated_sidecar_report(
                input_root=root / "data/pilot/annotated_sidecar_v1",
                output_dir=root / "data/pilot/annotated_sidecar_v1/reports",
                compare_aux_root=root / "data/pilot/language_probe_v4",
            )
            regenerated.append("annotated_sidecar_report")
        if not existence["retrieval_manifest"]:
            build_comment_retrieval_eval_regime(
                input_root=root / "data/pilot/annotated_sidecar_v1",
                output_root=root / "outputs/week10/comment_retrieval",
                config=CommentRetrievalEvalConfig(),
            )
            regenerated.append("retrieval_eval_manifest")
        if (
            not existence["retrieval_results_json"]
            or not existence["retrieval_results_csv"]
            or not existence["retrieval_summary_md"]
        ):
            run_raw_text_retrieval_probes(
                embedding_root=root / "outputs/week10/embedding_exports",
                corpus_root=root / "outputs/week10/comment_retrieval/probe_subset",
                output_dir=root / "outputs/week10/comment_retrieval",
                backbone_seeds=[11, 17, 23],
                mate_min_df=25,
                max_vocab_size=512,
                families=["annotated_sidecar"],
                output_prefix="comment_retrieval",
            )
            regenerated.append("comment_retrieval_results")

    existence = {name: path.exists() for name, path in paths.items()}
    missing = [name for name, exists in existence.items() if not exists]
    mismatches: list[str] = []
    suspicious_findings: list[str] = []

    annotated_report_json: dict[str, Any] | None = None
    if existence["annotated_report_json"]:
        annotated_report_json = _load_json(paths["annotated_report_json"])
        actual_report_json = generate_annotated_sidecar_report(
            input_root=root / "data/pilot/annotated_sidecar_v1",
            compare_aux_root=root / "data/pilot/language_probe_v4",
        )
        if annotated_report_json != actual_report_json:
            mismatches.append("annotated_sidecar_report_json_mismatch")
            if regenerate_if_stale:
                write_annotated_sidecar_report(
                    input_root=root / "data/pilot/annotated_sidecar_v1",
                    output_dir=root / "data/pilot/annotated_sidecar_v1/reports",
                    compare_aux_root=root / "data/pilot/language_probe_v4",
                )
                regenerated.append("annotated_sidecar_report_refresh")
                annotated_report_json = _load_json(paths["annotated_report_json"])
                if annotated_report_json == actual_report_json:
                    mismatches.pop()

    retrieval_manifest: dict[str, Any] | None = None
    if existence["retrieval_manifest"]:
        retrieval_manifest = _load_yaml(paths["retrieval_manifest"])
        for split_name in ("train", "val", "test"):
            split_payload = retrieval_manifest.get("splits", {}).get(split_name, {})
            output_path = Path(str(split_payload.get("output_path") or ""))
            if not output_path.is_absolute():
                output_path = root / output_path
            if not output_path.exists():
                mismatches.append(f"retrieval_subset_missing:{split_name}")
                continue
            if _jsonl_row_count(output_path) != int(split_payload.get("selected_rows", -1)):
                mismatches.append(f"retrieval_subset_count_mismatch:{split_name}")

    retrieval_results_json: dict[str, Any] | None = None
    if existence["retrieval_results_json"] and existence["retrieval_results_csv"]:
        retrieval_results_json = _load_json(paths["retrieval_results_json"])
        retrieval_results_csv = _load_csv_rows(paths["retrieval_results_csv"])
        if len(retrieval_results_json.get("results", [])) != len(retrieval_results_csv):
            mismatches.append("retrieval_json_csv_row_count_mismatch")
        aggregate_rows = retrieval_results_json.get("aggregate", [])
        if not aggregate_rows:
            mismatches.append("retrieval_aggregate_empty")
        for row in aggregate_rows:
            if math.isclose(
                float(row["board_to_text_recall_at_1_mean"]),
                float(row["board_to_text_recall_at_5_mean"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                suspicious_findings.append("board_to_text_r1_equals_r5_in_aggregate")
                break

    if annotated_report_json is not None and float(annotated_report_json["duplicate_position_comment_rate"]) >= 0.05:
        suspicious_findings.append("duplicate_position_comment_rate_above_5_percent")

    gate = choose_verification_gate(
        missing_count=len(missing),
        mismatch_count=len(mismatches),
        suspicious_findings=suspicious_findings,
    )
    verification = {
        "required_paths": {name: str(path) for name, path in paths.items()},
        "exists": existence,
        "regenerated": regenerated,
        "missing": missing,
        "mismatches": mismatches,
        "suspicious_findings": suspicious_findings,
        "gate_result": gate,
        "annotated_sidecar": annotated_report_json,
        "retrieval_manifest": retrieval_manifest,
        "retrieval_summary": {
            "result_count": len(retrieval_results_json.get("results", [])) if retrieval_results_json else 0,
            "aggregate_count": len(retrieval_results_json.get("aggregate", [])) if retrieval_results_json else 0,
            "best_board_to_text": (
                max(
                    retrieval_results_json.get("aggregate", []),
                    key=lambda row: float(row["board_to_text_mrr_mean"]),
                )
                if retrieval_results_json and retrieval_results_json.get("aggregate")
                else None
            ),
            "best_text_to_board": (
                max(
                    retrieval_results_json.get("aggregate", []),
                    key=lambda row: float(row["text_to_board_mrr_mean"]),
                )
                if retrieval_results_json and retrieval_results_json.get("aggregate")
                else None
            ),
        },
    }

    json_path = output_root / "week10_artifact_verification.json"
    md_path = output_root / "week10_artifact_verification.md"
    json_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")

    lines = ["# Week-10 Artifact Verification", ""]
    lines.append(f"- gate_result: `{gate}`")
    lines.append(f"- regenerated: {', '.join(regenerated) if regenerated else 'none'}")
    lines.append(f"- missing: {', '.join(missing) if missing else 'none'}")
    lines.append(f"- mismatches: {', '.join(mismatches) if mismatches else 'none'}")
    lines.append(f"- suspicious_findings: {', '.join(suspicious_findings) if suspicious_findings else 'none'}")
    if annotated_report_json:
        lines.append("")
        lines.append("## Annotated Sidecar")
        for split_name, count in annotated_report_json["total_rows_by_split"].items():
            unique_games = annotated_report_json["unique_game_ids_by_split"][split_name]
            lines.append(f"- `{split_name}`: rows={count}, unique_games={unique_games}")
        lines.append(f"- valid_target_move_uci: {annotated_report_json['rows_with_valid_target_move_uci']}")
        lines.append(f"- valid_next_fen: {annotated_report_json['rows_with_valid_next_fen']}")
        lines.append(
            f"- duplicate_position_comment_rate: {float(annotated_report_json['duplicate_position_comment_rate']):.6f}"
        )
    if retrieval_manifest:
        lines.append("")
        lines.append("## Retrieval Regime")
        lines.append(f"- evaluation_mode: {retrieval_manifest['evaluation_mode']}")
        lines.append(f"- sampling_rule: {retrieval_manifest['sampling_rule']}")
        for split_name, payload in retrieval_manifest["splits"].items():
            lines.append(
                f"- `{split_name}`: eligible={payload['eligible_rows']}, selected={payload['selected_rows']}"
            )
    best_board = verification["retrieval_summary"]["best_board_to_text"]
    best_text = verification["retrieval_summary"]["best_text_to_board"]
    if best_board:
        lines.append("")
        lines.append("## Best Retrieval Rows")
        lines.append(
            f"- board_to_text: backbone={best_board['backbone']}, pool={best_board['pool']}, probe={best_board['probe_model']}, "
            f"R@1={float(best_board['board_to_text_recall_at_1_mean']):.4f}, "
            f"R@5={float(best_board['board_to_text_recall_at_5_mean']):.4f}, "
            f"MRR={float(best_board['board_to_text_mrr_mean']):.4f}"
        )
    if best_text:
        lines.append(
            f"- text_to_board: backbone={best_text['backbone']}, pool={best_text['pool']}, probe={best_text['probe_model']}, "
            f"R@1={float(best_text['text_to_board_recall_at_1_mean']):.4f}, "
            f"R@5={float(best_text['text_to_board_recall_at_5_mean']):.4f}, "
            f"MRR={float(best_text['text_to_board_mrr_mean']):.4f}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    verification["json_path"] = str(json_path)
    verification["md_path"] = str(md_path)
    return verification
