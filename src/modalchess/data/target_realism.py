"""Target-realism reporting and conservative MATE v2 target materialization."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import yaml

from modalchess.data.preprocessing_common import iter_records_from_path, write_jsonl
from modalchess.data.probe_targets import derive_mate_target_payload


def _load_json(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        return {}
    return json.loads(path_obj.read_text(encoding="utf-8"))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        return {}
    payload = yaml.safe_load(path_obj.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _selected_v2_candidates(audit_report: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in audit_report.get("candidate_labels", []):
        if not row.get("recommended_for_future_v2"):
            continue
        if str(row.get("ambiguity_risk")) == "high":
            continue
        selected.append(dict(row))
    return selected


def _extra_keyword_map(candidate_rows: list[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    return {
        str(row["label"]): tuple(str(pattern) for pattern in row.get("patterns", []))
        for row in candidate_rows
    }


def _build_mate_targets_v2(
    *,
    probe_root: Path,
    output_root: Path,
    extra_keyword_map: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    outputs: dict[str, str] = {}
    label_counter: Counter[str] = Counter()
    counts_by_split: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        input_path = probe_root / f"mate_{split_name}.jsonl"
        rows = [dict(row) for row in iter_records_from_path(input_path)]
        target_rows = [derive_mate_target_payload(row, keyword_map=extra_keyword_map) for row in rows]
        for row in target_rows:
            for label_name in row.get("target_labels", []):
                label_counter[str(label_name)] += 1
        output_path = output_root / f"mate_targets_v2_{split_name}.jsonl"
        write_jsonl(output_path, target_rows)
        outputs[split_name] = str(output_path)
        counts_by_split[split_name] = len(target_rows)
    return {
        "outputs": outputs,
        "counts_by_split": counts_by_split,
        "label_counts": dict(sorted(label_counter.items())),
    }


def write_target_realism_report(
    *,
    probe_root: str | Path = "data/pilot/language_probe_v3_fix",
    aux_root: str | Path = "data/pilot/language_probe_v4",
    mate_keyword_audit_path: str | Path = "data/pilot/language_probe_v3/reports/mate_keyword_audit.json",
    output_root: str | Path = "data/pilot/language_probe_v4",
    create_mate_v2: bool = True,
) -> dict[str, Any]:
    """Write week-8 target-realism report and optional MATE v2 targets."""
    probe_root_path = Path(probe_root)
    aux_root_path = Path(aux_root)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    report_dir = output_root_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    keyword_audit = _load_json(mate_keyword_audit_path)
    selected_candidates = _selected_v2_candidates(keyword_audit)
    extra_keyword_map = _extra_keyword_map(selected_candidates)
    probe_manifest = _load_yaml(probe_root_path / "manifests" / "probe_targets_manifest.yaml")
    aux_manifest = _load_yaml(aux_root_path / "manifests" / "aux_source_manifest.yaml")

    mate_v2_payload: dict[str, Any] = {
        "outputs": {},
        "counts_by_split": {},
        "label_counts": {},
    }
    if create_mate_v2 and selected_candidates:
        mate_v2_payload = _build_mate_targets_v2(
            probe_root=probe_root_path,
            output_root=output_root_path,
            extra_keyword_map=extra_keyword_map,
        )

    aux_board_anchored_total = sum(
        int(value) for value in aux_manifest.get("board_anchored_split_counts", {}).values()
    )
    conclusions: list[str] = []
    if selected_candidates:
        conclusions.append(
            "Conservative MATE v2 labels are justified for trade_up, open_line_attack, sacrifice_for_attack, and king_attack_zone."
        )
    else:
        conclusions.append("No conservative MATE v2 label expansion was justified.")
    if aux_board_anchored_total > 2:
        conclusions.append("Auxiliary board-anchored text coverage is materially larger than the week-7 sample-only state.")
    else:
        conclusions.append("Auxiliary board-anchored text coverage is still too small to count as a material realism upgrade.")

    report = {
        "probe_root": str(probe_root_path),
        "aux_root": str(aux_root_path),
        "selected_v2_labels": [row["label"] for row in selected_candidates],
        "selected_v2_label_support": {
            str(row["label"]): int(row["support_count"]) for row in selected_candidates
        },
        "created_mate_targets_v2": bool(mate_v2_payload["outputs"]),
        "v1_label_counts": probe_manifest.get("label_frequencies_by_source", {}).get("mate", {}),
        "v2_label_counts": mate_v2_payload["label_counts"],
        "v2_counts_by_split": mate_v2_payload["counts_by_split"],
        "v2_outputs": mate_v2_payload["outputs"],
        "aux_board_anchored_rows": aux_board_anchored_total,
        "aux_text_only_rows": int(aux_manifest.get("text_only_count", 0)),
        "conclusions": conclusions,
    }
    json_path = report_dir / "target_realism_report.json"
    md_path = report_dir / "target_realism_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "report": report,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Target Realism Report", ""]
    lines.append(f"- created_mate_targets_v2: {report['created_mate_targets_v2']}")
    lines.append(f"- aux_board_anchored_rows: {report['aux_board_anchored_rows']}")
    lines.append(f"- aux_text_only_rows: {report['aux_text_only_rows']}")
    lines.append("")
    lines.append("## Selected MATE v2 Labels")
    if report["selected_v2_labels"]:
        for label_name in report["selected_v2_labels"]:
            lines.append(
                f"- `{label_name}`: support={report['selected_v2_label_support'][label_name]}"
            )
    else:
        lines.append("- No v2 labels selected.")
    lines.append("")
    lines.append("## Conclusions")
    for conclusion in report["conclusions"]:
        lines.append(f"- {conclusion}")
    return "\n".join(lines) + "\n"
