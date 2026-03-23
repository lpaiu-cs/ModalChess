"""Build shared comparable source-holdout regimes across multiple variants."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from modalchess.data.preprocessing_common import iter_records_from_path, write_jsonl, write_yaml
from modalchess.data.source_holdout_eval import (
    HoldoutThresholds,
    SourceHoldoutEvalConfig,
    build_source_holdout_eval,
)


@dataclass(slots=True)
class SharedSourceHoldoutEvalConfig:
    """Configuration for week-18 shared source-holdout construction."""

    coarse_thresholds: HoldoutThresholds = field(
        default_factory=lambda: HoldoutThresholds(
            min_total_rows=300,
            min_train_rows=150,
            min_val_rows=20,
            min_test_rows=20,
        )
    )
    family_thresholds: HoldoutThresholds = field(
        default_factory=lambda: HoldoutThresholds(
            min_total_rows=300,
            min_train_rows=150,
            min_val_rows=20,
            min_test_rows=20,
        )
    )
    min_shared_test_rows: int = 100
    categories: tuple[str, ...] = (
        "mixed_baseline",
        "coarse_source_holdout",
        "source_family_holdout",
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _regime_key(regime: Mapping[str, Any]) -> tuple[str, str | None, str | None]:
    category = str(regime.get("category") or "unknown")
    holdout_field = regime.get("holdout_field")
    holdout_value = regime.get("holdout_value")
    if category == "mixed_baseline":
        return ("mixed_baseline", None, None)
    return (
        category,
        str(holdout_field) if holdout_field is not None else None,
        str(holdout_value) if holdout_value is not None else None,
    )


def _probe_id(row: Mapping[str, Any]) -> str:
    return str(row.get("probe_id") or row.get("sidecar_id") or "")


def _load_rows_by_split(regime_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        split_name: [
            dict(row)
            for row in iter_records_from_path(regime_dir / f"annotated_sidecar_{split_name}.jsonl")
        ]
        for split_name in ("train", "val", "test")
    }


def _filter_test_rows(
    rows_by_split: Mapping[str, list[dict[str, Any]]],
    shared_test_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": [dict(row) for row in rows_by_split["train"]],
        "val": [dict(row) for row in rows_by_split["val"]],
        "test": [
            dict(row)
            for row in rows_by_split["test"]
            if _probe_id(row) in shared_test_ids
        ],
    }


def _variant_manifest_payload(
    *,
    input_root: str,
    config: SharedSourceHoldoutEvalConfig,
    regimes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "input_root": input_root,
        "config": {
            "coarse_thresholds": asdict(config.coarse_thresholds),
            "family_thresholds": asdict(config.family_thresholds),
            "min_shared_test_rows": config.min_shared_test_rows,
            "categories": list(config.categories),
        },
        "regimes": regimes,
        "source_types": {},
    }


def build_shared_source_holdout_eval(
    *,
    variant_roots: Mapping[str, str | Path],
    output_root: str | Path = "data/pilot/annotated_sidecar_holdout_v2",
    config: SharedSourceHoldoutEvalConfig | None = None,
) -> dict[str, Any]:
    """Build variant-specific holdouts with shared comparable test pools."""
    shared_config = config or SharedSourceHoldoutEvalConfig()
    output_path = Path(output_root)
    raw_root = output_path / "_raw"
    variants_root = output_path / "variants"
    shared_ids_root = output_path / "shared_test_ids"
    reports_dir = output_path / "reports"
    manifests_dir = output_path / "manifests"
    raw_root.mkdir(parents=True, exist_ok=True)
    variants_root.mkdir(parents=True, exist_ok=True)
    shared_ids_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    variant_manifests: dict[str, dict[str, Any]] = {}
    variant_regimes_by_key: dict[str, dict[tuple[str, str | None, str | None], dict[str, Any]]] = {}

    for variant_name, input_root in variant_roots.items():
        raw_variant_root = raw_root / variant_name
        build_source_holdout_eval(
            input_root=input_root,
            output_root=raw_variant_root,
            config=SourceHoldoutEvalConfig(
                coarse_thresholds=shared_config.coarse_thresholds,
                family_thresholds=shared_config.family_thresholds,
            ),
        )
        manifest = _load_yaml(raw_variant_root / "manifests" / "holdout_manifest.yaml")
        variant_manifests[variant_name] = manifest
        regimes_by_key: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}
        for regime in manifest.get("regimes", []):
            key = _regime_key(regime)
            if key[0] not in shared_config.categories:
                continue
            regimes_by_key[key] = dict(regime)
        variant_regimes_by_key[variant_name] = regimes_by_key

    common_keys = sorted(
        set.intersection(*(set(regimes.keys()) for regimes in variant_regimes_by_key.values()))
    )

    shared_regimes: list[dict[str, Any]] = []
    excluded_regimes: list[dict[str, Any]] = []
    per_variant_regimes: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for key in common_keys:
        category, holdout_field, holdout_value = key
        regime_rows_by_variant: dict[str, dict[str, list[dict[str, Any]]]] = {}
        test_probe_sets: list[set[str]] = []
        for variant_name, regimes_by_key in variant_regimes_by_key.items():
            regime = regimes_by_key[key]
            regime_dir = Path(str(regime["regime_dir"]))
            rows_by_split = _load_rows_by_split(regime_dir)
            regime_rows_by_variant[variant_name] = rows_by_split
            test_probe_sets.append({_probe_id(row) for row in rows_by_split["test"]})

        shared_test_ids = set.intersection(*test_probe_sets)
        shared_regime_slug = _slugify(
            "mixed_baseline" if category == "mixed_baseline" else f"{category}_{holdout_value}"
        )
        shared_ids_path = shared_ids_root / f"{shared_regime_slug}.json"
        shared_ids_path.write_text(
            json.dumps(
                {
                    "category": category,
                    "holdout_field": holdout_field,
                    "holdout_value": holdout_value,
                    "shared_test_probe_ids": sorted(shared_test_ids),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if len(shared_test_ids) < shared_config.min_shared_test_rows:
            excluded_regimes.append(
                {
                    "category": category,
                    "holdout_field": holdout_field,
                    "holdout_value": holdout_value,
                    "shared_test_rows": len(shared_test_ids),
                    "reason": f"shared_test_rows<{shared_config.min_shared_test_rows}",
                    "shared_test_ids_path": str(shared_ids_path),
                }
            )
            continue

        shared_summary = {
            "category": category,
            "holdout_field": holdout_field,
            "holdout_value": holdout_value,
            "shared_test_rows": len(shared_test_ids),
            "shared_test_ids_path": str(shared_ids_path),
        }
        shared_regimes.append(shared_summary)

        for variant_name, rows_by_split in regime_rows_by_variant.items():
            raw_regime = variant_regimes_by_key[variant_name][key]
            filtered_rows = _filter_test_rows(rows_by_split, shared_test_ids)
            final_regime_dir = variants_root / variant_name / "regimes" / str(raw_regime["regime_name"])
            split_counts = {
                split_name: write_jsonl(
                    final_regime_dir / f"annotated_sidecar_{split_name}.jsonl",
                    filtered_rows[split_name],
                )
                for split_name in ("train", "val", "test")
            }
            per_variant_regimes[variant_name].append(
                {
                    "regime_name": raw_regime["regime_name"],
                    "category": raw_regime["category"],
                    "holdout_field": raw_regime.get("holdout_field"),
                    "holdout_value": raw_regime.get("holdout_value"),
                    "regime_dir": str(final_regime_dir),
                    "split_counts": split_counts,
                    "shared_test_rows": len(shared_test_ids),
                    "shared_test_ids_path": str(shared_ids_path),
                }
            )

    variant_manifest_paths: dict[str, str] = {}
    for variant_name, regimes in per_variant_regimes.items():
        variant_root = variants_root / variant_name
        variant_reports_dir = variant_root / "reports"
        variant_manifests_dir = variant_root / "manifests"
        variant_reports_dir.mkdir(parents=True, exist_ok=True)
        variant_manifests_dir.mkdir(parents=True, exist_ok=True)
        variant_manifest = _variant_manifest_payload(
            input_root=str(variant_roots[variant_name]),
            config=shared_config,
            regimes=sorted(regimes, key=lambda item: (str(item["category"]), str(item["regime_name"]))),
        )
        manifest_path = variant_manifests_dir / "holdout_manifest.yaml"
        write_yaml(manifest_path, variant_manifest)
        variant_manifest_paths[variant_name] = str(manifest_path)
        variant_report = {
            "variant_name": variant_name,
            "input_root": str(variant_roots[variant_name]),
            "shared_regime_count": len(regimes),
            "regimes": variant_manifest["regimes"],
        }
        (variant_reports_dir / "holdout_design_report.json").write_text(
            json.dumps(variant_report, indent=2),
            encoding="utf-8",
        )
        lines = [
            "# Shared Source Holdout Design Report",
            "",
            f"- variant: `{variant_name}`",
            f"- input_root: `{variant_roots[variant_name]}`",
            f"- shared_regime_count: `{len(regimes)}`",
            "",
        ]
        for regime in variant_manifest["regimes"]:
            lines.append(
                f"- `{regime['regime_name']}` ({regime['category']}): split_counts={regime['split_counts']}, "
                f"shared_test_rows={regime['shared_test_rows']}"
            )
        (variant_reports_dir / "holdout_design_report.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    top_level_manifest = {
        "variant_roots": {key: str(value) for key, value in variant_roots.items()},
        "config": {
            "coarse_thresholds": asdict(shared_config.coarse_thresholds),
            "family_thresholds": asdict(shared_config.family_thresholds),
            "min_shared_test_rows": shared_config.min_shared_test_rows,
            "categories": list(shared_config.categories),
        },
        "shared_regimes": shared_regimes,
        "excluded_regimes": excluded_regimes,
        "variant_manifest_paths": variant_manifest_paths,
    }
    manifest_path = manifests_dir / "holdout_manifest.yaml"
    write_yaml(manifest_path, top_level_manifest)

    report = {
        "variant_roots": {key: str(value) for key, value in variant_roots.items()},
        "shared_regimes": shared_regimes,
        "excluded_regimes": excluded_regimes,
        "variant_shared_regime_counts": {
            variant_name: len(regimes)
            for variant_name, regimes in per_variant_regimes.items()
        },
    }
    report_json_path = reports_dir / "holdout_design_report.json"
    report_md_path = reports_dir / "holdout_design_report.md"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Shared Source Holdout Design Report",
        "",
        f"- shared_regime_count: `{len(shared_regimes)}`",
        f"- excluded_regime_count: `{len(excluded_regimes)}`",
        "",
        "## Shared Regimes",
    ]
    for regime in shared_regimes:
        lines.append(
            f"- `{regime['category']}` / `{regime['holdout_value'] or 'mixed_baseline'}`: "
            f"shared_test_rows={regime['shared_test_rows']}"
        )
    if excluded_regimes:
        lines.append("")
        lines.append("## Excluded Regimes")
        for regime in excluded_regimes:
            lines.append(
                f"- `{regime['category']}` / `{regime['holdout_value'] or 'mixed_baseline'}`: "
                f"{regime['reason']} ({regime['shared_test_rows']})"
            )
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "manifest_path": str(manifest_path),
        "report_json": str(report_json_path),
        "report_md": str(report_md_path),
        "shared_regime_count": len(shared_regimes),
        "excluded_regime_count": len(excluded_regimes),
        "variant_manifest_paths": variant_manifest_paths,
    }
