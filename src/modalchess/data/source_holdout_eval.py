"""Source-holdout evaluation regimes for multisource annotated comment retrieval."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable

from modalchess.data.preprocessing_common import iter_records_from_path, write_jsonl, write_yaml


@dataclass(slots=True)
class HoldoutThresholds:
    min_total_rows: int
    min_train_rows: int
    min_val_rows: int
    min_test_rows: int


@dataclass(slots=True)
class SourceHoldoutEvalConfig:
    coarse_thresholds: HoldoutThresholds = field(
        default_factory=lambda: HoldoutThresholds(
            min_total_rows=1000,
            min_train_rows=1000,
            min_val_rows=100,
            min_test_rows=100,
        )
    )
    family_thresholds: HoldoutThresholds = field(
        default_factory=lambda: HoldoutThresholds(
            min_total_rows=1500,
            min_train_rows=1000,
            min_val_rows=100,
            min_test_rows=100,
        )
    )
    holdout_test_fraction_warn: float = 0.85


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def _load_rows_by_split(root: str | Path) -> dict[str, list[dict[str, Any]]]:
    input_root = Path(root)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "val", "test"):
        rows_by_split[split_name] = [
            dict(row)
            for row in iter_records_from_path(input_root / f"{split_name}.jsonl")
        ]
    return rows_by_split


def _counts_by_group(rows_by_split: dict[str, list[dict[str, Any]]], field_name: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for split_name, rows in rows_by_split.items():
        for row in rows:
            group_name = str(row.get(field_name) or "unknown")
            counts[group_name][split_name] += 1
    return dict(sorted(counts.items(), key=lambda item: (-(sum(item[1].values())), item[0])))


def _source_family_proportions(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    counter = Counter(str(row.get("source_family") or "unknown") for row in rows)
    total = sum(counter.values())
    return {
        family: (count / total) if total else 0.0
        for family, count in sorted(counter.items())
    }


def _source_type_label(row: dict[str, Any]) -> str:
    source_name = str(row.get("source") or "")
    if source_name == "waterhorse_annotated_pgn":
        return "waterhorse_only"
    if source_name.startswith("mate_"):
        return "mate_only"
    return "other"


def _group_eligibility_reason(counts: dict[str, int], thresholds: HoldoutThresholds) -> str | None:
    total_rows = sum(counts.values())
    if total_rows < thresholds.min_total_rows:
        return f"total<{thresholds.min_total_rows}"
    if counts["train"] < thresholds.min_train_rows:
        return f"train<{thresholds.min_train_rows}"
    if counts["val"] < thresholds.min_val_rows:
        return f"val<{thresholds.min_val_rows}"
    if counts["test"] < thresholds.min_test_rows:
        return f"test<{thresholds.min_test_rows}"
    return None


def _write_regime_dir(regime_dir: Path, rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    regime_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        counts[split_name] = write_jsonl(
            regime_dir / f"annotated_sidecar_{split_name}.jsonl",
            rows_by_split[split_name],
        )
    return counts


def _regime_metadata(
    *,
    regime_name: str,
    category: str,
    holdout_field: str | None,
    holdout_value: str | None,
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    return {
        "regime_name": regime_name,
        "category": category,
        "holdout_field": holdout_field,
        "holdout_value": holdout_value,
        "split_counts": {split_name: len(rows) for split_name, rows in rows_by_split.items()},
        "source_counts": dict(
            Counter(str(row.get("source") or "unknown") for row in all_rows).most_common()
        ),
        "source_family_counts": dict(
            Counter(str(row.get("source_family") or "unknown") for row in all_rows).most_common()
        ),
        "comment_source_counts": dict(
            Counter(str(row.get("comment_source") or "unknown") for row in all_rows).most_common()
        ),
        "source_family_proportions": _source_family_proportions(all_rows),
    }


def _copy_full_regime(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {
        split_name: [dict(row) for row in rows]
        for split_name, rows in rows_by_split.items()
    }


def _holdout_regime_rows(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    field_name: str,
    holdout_value: str,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": [
            dict(row)
            for row in rows_by_split["train"]
            if str(row.get(field_name) or "unknown") != holdout_value
        ],
        "val": [
            dict(row)
            for row in rows_by_split["val"]
            if str(row.get(field_name) or "unknown") != holdout_value
        ],
        "test": [
            dict(row)
            for row in rows_by_split["test"]
            if str(row.get(field_name) or "unknown") == holdout_value
        ],
    }


def _source_type_regime_rows(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    source_type: str,
) -> dict[str, list[dict[str, Any]]]:
    if source_type == "mixed":
        return _copy_full_regime(rows_by_split)
    return {
        split_name: [
            dict(row)
            for row in rows
            if _source_type_label(row) == source_type
        ]
        for split_name, rows in rows_by_split.items()
    }


def build_source_holdout_eval(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_eval_v5",
    output_root: str | Path = "data/pilot/annotated_sidecar_holdout_v1",
    config: SourceHoldoutEvalConfig | None = None,
) -> dict[str, Any]:
    holdout_config = config or SourceHoldoutEvalConfig()
    input_path = Path(input_root)
    output_path = Path(output_root)
    reports_dir = output_path / "reports"
    manifests_dir = output_path / "manifests"
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = _load_rows_by_split(input_path)
    source_counts = _counts_by_group(rows_by_split, "source")
    family_counts = _counts_by_group(rows_by_split, "source_family")

    coarse_eligible: dict[str, dict[str, int]] = {}
    coarse_excluded: dict[str, dict[str, Any]] = {}
    for source_name, counts in source_counts.items():
        reason = _group_eligibility_reason(counts, holdout_config.coarse_thresholds)
        if reason is None:
            coarse_eligible[source_name] = counts
        else:
            coarse_excluded[source_name] = {"counts": counts, "reason": reason}

    family_eligible: dict[str, dict[str, int]] = {}
    family_excluded: dict[str, dict[str, Any]] = {}
    for family_name, counts in family_counts.items():
        reason = _group_eligibility_reason(counts, holdout_config.family_thresholds)
        if reason is None:
            family_eligible[family_name] = counts
        else:
            family_excluded[family_name] = {"counts": counts, "reason": reason}

    regimes: list[dict[str, Any]] = []

    mixed_dir = output_path / "mixed_baseline"
    mixed_rows = _copy_full_regime(rows_by_split)
    mixed_counts = _write_regime_dir(mixed_dir, mixed_rows)
    regimes.append(
        {
            **_regime_metadata(
                regime_name="mixed_baseline",
                category="mixed_baseline",
                holdout_field=None,
                holdout_value=None,
                rows_by_split=mixed_rows,
            ),
            "regime_dir": str(mixed_dir),
            "split_counts": mixed_counts,
        }
    )

    for source_name, counts in coarse_eligible.items():
        regime_name = f"coarse_source_holdout__{_slugify(source_name)}"
        regime_dir = output_path / "coarse_source_holdouts" / _slugify(source_name)
        regime_rows = _holdout_regime_rows(rows_by_split, field_name="source", holdout_value=source_name)
        split_counts = _write_regime_dir(regime_dir, regime_rows)
        regimes.append(
            {
                **_regime_metadata(
                    regime_name=regime_name,
                    category="coarse_source_holdout",
                    holdout_field="source",
                    holdout_value=source_name,
                    rows_by_split=regime_rows,
                ),
                "heldout_counts": counts,
                "regime_dir": str(regime_dir),
                "split_counts": split_counts,
            }
        )

    for family_name, counts in family_eligible.items():
        regime_name = f"source_family_holdout__{_slugify(family_name)}"
        regime_dir = output_path / "source_family_holdouts" / _slugify(family_name)
        regime_rows = _holdout_regime_rows(rows_by_split, field_name="source_family", holdout_value=family_name)
        split_counts = _write_regime_dir(regime_dir, regime_rows)
        regimes.append(
            {
                **_regime_metadata(
                    regime_name=regime_name,
                    category="source_family_holdout",
                    holdout_field="source_family",
                    holdout_value=family_name,
                    rows_by_split=regime_rows,
                ),
                "heldout_counts": counts,
                "regime_dir": str(regime_dir),
                "split_counts": split_counts,
            }
        )

    source_types: dict[str, dict[str, Any]] = {}
    for source_type in ("mixed", "waterhorse_only", "mate_only"):
        regime_name = f"source_type__{source_type}"
        regime_dir = output_path / "source_types" / source_type
        regime_rows = _source_type_regime_rows(rows_by_split, source_type=source_type)
        split_counts = _write_regime_dir(regime_dir, regime_rows)
        source_types[source_type] = {
            **_regime_metadata(
                regime_name=regime_name,
                category="source_type_ablation",
                holdout_field="source_type",
                holdout_value=source_type,
                rows_by_split=regime_rows,
            ),
            "regime_dir": str(regime_dir),
            "split_counts": split_counts,
        }

    manifest = {
        "input_root": str(input_path),
        "config": {
            "coarse_thresholds": asdict(holdout_config.coarse_thresholds),
            "family_thresholds": asdict(holdout_config.family_thresholds),
            "holdout_test_fraction_warn": holdout_config.holdout_test_fraction_warn,
        },
        "input_split_counts": {split_name: len(rows) for split_name, rows in rows_by_split.items()},
        "coarse_source_eligible": coarse_eligible,
        "coarse_source_excluded": coarse_excluded,
        "source_family_eligible": family_eligible,
        "source_family_excluded": family_excluded,
        "regimes": regimes,
        "source_types": source_types,
    }
    manifest_path = manifests_dir / "holdout_manifest.yaml"
    write_yaml(manifest_path, manifest)

    report = {
        "input_root": str(input_path),
        "input_split_counts": {split_name: len(rows) for split_name, rows in rows_by_split.items()},
        "coarse_source_eligible": coarse_eligible,
        "coarse_source_excluded": coarse_excluded,
        "source_family_eligible": family_eligible,
        "source_family_excluded": family_excluded,
        "regimes": regimes,
        "source_types": source_types,
    }
    report_json_path = reports_dir / "holdout_design_report.json"
    report_md_path = reports_dir / "holdout_design_report.md"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Source Holdout Design Report", ""]
    lines.append(f"- input_split_counts: {report['input_split_counts']}")
    lines.append(f"- coarse_source_eligible: {coarse_eligible}")
    lines.append(f"- coarse_source_excluded: {coarse_excluded}")
    lines.append(f"- source_family_eligible: {family_eligible}")
    lines.append(f"- source_family_excluded: {family_excluded}")
    lines.append("")
    lines.append("## Regimes")
    for regime in regimes:
        lines.append(
            f"- `{regime['regime_name']}` ({regime['category']}): split_counts={regime['split_counts']}"
        )
    lines.append("")
    lines.append("## Source Types")
    for source_type, payload in source_types.items():
        lines.append(f"- `{source_type}`: split_counts={payload['split_counts']}")
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "manifest_path": str(manifest_path),
        "report_json": str(report_json_path),
        "report_md": str(report_md_path),
        "regime_count": len(regimes),
        "source_types": source_types,
    }
