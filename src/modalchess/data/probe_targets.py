"""Conservative target derivation for week-5 probe corpora."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from modalchess.data.preprocessing_common import iter_records_from_path, special_rule_flags, write_jsonl, write_yaml


MATE_KEYWORD_MAP: dict[str, tuple[str, ...]] = {
    "check": (" check", "check ", "checking", "in check"),
    "mate_threat": ("checkmate", " mate", "mate!", "mate threat"),
    "fork": ("fork",),
    "pin": (" pin", "pinned"),
    "skewer": ("skewer",),
    "discovered_attack": ("discovered attack",),
    "discovered_check": ("discovered check",),
    "king_safety": ("king safety", "opposing king", "enemy king", "near the opposing king"),
    "passed_pawn": ("passed pawn",),
    "open_file": ("open file",),
    "promotion": ("promotion", "promote", "queen the pawn"),
    "capture": ("capture", "captures", "captured", "taken", "take ", "trades the lower value piece"),
    "recapture": ("recapture",),
}


@dataclass(slots=True)
class ProbeTargetConfig:
    """Target-derivation configuration."""

    rare_label_threshold: int = 20
    drop_rare_labels_from_output: bool = False


def merge_mate_keyword_maps(
    extra_keyword_map: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Merge the default MATE keyword map with conservative extensions."""
    merged = dict(MATE_KEYWORD_MAP)
    if extra_keyword_map is None:
        return merged
    for label_name, keywords in extra_keyword_map.items():
        merged[label_name] = tuple(dict.fromkeys((*merged.get(label_name, ()), *keywords)))
    return merged


def derive_mate_target_payload(
    row: dict[str, Any],
    keyword_map: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Derive conservative multi-label targets from MATE text."""
    text = " ".join(
        str(value or "")
        for value in (row.get("strategy_text"), row.get("tactic_text"))
    ).lower()
    labels: list[str] = []
    keyword_hits: dict[str, list[str]] = {}
    active_keyword_map = merge_mate_keyword_maps(keyword_map)
    for label_name, keywords in active_keyword_map.items():
        hits = [keyword for keyword in keywords if keyword in text]
        if hits:
            labels.append(label_name)
            keyword_hits[label_name] = hits
    return {
        "probe_id": row["probe_id"],
        "source": row["source"],
        "split": row["split"],
        "fen": row["fen"],
        "target_move_uci": row.get("target_move_uci"),
        "target_labels": labels,
        "keyword_hits": keyword_hits,
    }


def derive_puzzle_target_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Use theme tags directly and add special-rule flags."""
    flags = special_rule_flags(
        str(row["fen"]),
        str(row.get("target_move_uci")) if row.get("target_move_uci") else None,
    )
    return {
        "probe_id": row["probe_id"],
        "source": row["source"],
        "split": row["split"],
        "fen": row["fen"],
        "target_move_uci": row.get("target_move_uci"),
        "target_labels": list(row.get("theme_tags") or []),
        "promotion_flag": bool(flags["promotion"]),
        "castling_flag": bool(flags["castling"]),
        "en_passant_flag": bool(flags["en_passant"]),
        "check_evasion_flag": bool(flags["check_evasion"]),
    }


def build_probe_targets(
    *,
    input_root: str | Path,
    output_root: str | Path,
    config: ProbeTargetConfig | None = None,
) -> dict[str, Any]:
    """Write target JSONL files for standalone probe corpora."""
    target_config = config or ProbeTargetConfig()
    input_dir = Path(input_root)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, Any] = {}
    outputs: dict[str, str] = {}
    label_counts_by_source: dict[str, dict[str, int]] = {}
    rare_labels_by_source: dict[str, list[str]] = {}
    empty_text_rates_by_source: dict[str, float] = {}
    labels_removed_by_source: dict[str, list[str]] = {}
    warnings: list[str] = []

    for source_name, derive_fn in (("mate", derive_mate_target_payload), ("puzzle", derive_puzzle_target_payload)):
        label_counter: Counter[str] = Counter()
        split_counts: dict[str, int] = {}
        split_target_rows: dict[str, list[dict[str, Any]]] = {}
        source_rows: list[dict[str, Any]] = []
        for split_name in ("train", "val", "test"):
            input_path = input_dir / f"{source_name}_{split_name}.jsonl"
            rows = [dict(row) for row in iter_records_from_path(input_path)]
            source_rows.extend(rows)
            target_rows = [derive_fn(row) for row in rows]
            split_target_rows[split_name] = target_rows
            for target_row in target_rows:
                for label in target_row.get("target_labels", []):
                    label_counter[str(label)] += 1

        rare_labels = sorted([label_name for label_name, label_count in label_counter.items() if label_count < target_config.rare_label_threshold])
        rare_label_set = set(rare_labels)
        rare_labels_by_source[source_name] = rare_labels
        labels_removed_by_source[source_name] = rare_labels if target_config.drop_rare_labels_from_output else []
        empty_text_count = sum(
            int(not (row.get("strategy_text") or row.get("tactic_text")))
            for row in source_rows
        )
        empty_text_rates_by_source[source_name] = (empty_text_count / len(source_rows)) if source_rows else 1.0

        for split_name in ("train", "val", "test"):
            target_rows = split_target_rows[split_name]
            if target_config.drop_rare_labels_from_output and rare_labels:
                target_rows = [
                    {
                        **target_row,
                        "target_labels": [
                            str(label)
                            for label in target_row.get("target_labels", [])
                            if str(label) not in rare_label_set
                        ],
                    }
                    for target_row in target_rows
                ]
            output_path = output_dir / f"{source_name}_targets_{split_name}.jsonl"
            write_jsonl(output_path, target_rows)
            outputs[f"{source_name}_{split_name}"] = str(output_path)
            split_counts[split_name] = len(target_rows)
        label_counts_by_source[source_name] = dict(sorted(label_counter.items()))
        counts[source_name] = split_counts
        for label_name in rare_labels:
            warnings.append(f"{source_name} label `{label_name}` is rare ({label_counter[label_name]}).")

    manifest_path = output_dir / "manifests" / "probe_targets_manifest.yaml"
    write_yaml(
        manifest_path,
        {
            "config": asdict(target_config),
            "counts": counts,
            "label_frequencies_by_source": label_counts_by_source,
            "rare_labels_by_source": rare_labels_by_source,
            "empty_text_rates_by_source": empty_text_rates_by_source,
            "labels_removed_by_rarity_threshold": labels_removed_by_source,
            "warnings": warnings,
            "outputs": outputs,
        },
    )
    return {
        "manifest_path": str(manifest_path),
        "counts": counts,
        "label_counts_by_source": label_counts_by_source,
        "rare_labels_by_source": rare_labels_by_source,
        "empty_text_rates_by_source": empty_text_rates_by_source,
        "labels_removed_by_rarity_threshold": labels_removed_by_source,
        "warnings": warnings,
        "outputs": outputs,
    }
