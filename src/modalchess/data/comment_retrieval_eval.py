"""Build a reproducible evaluation regime for annotated comment retrieval."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from typing import Any

from modalchess.data.preprocessing_common import iter_records_from_path, write_jsonl, write_yaml


@dataclass(slots=True)
class CommentRetrievalEvalConfig:
    """Configuration for deterministic comment-retrieval evaluation subsets."""

    train_limit: int = 100000
    val_limit: int = 5000
    test_limit: int = 5000
    salt: str = "modalchess_week10_comment_eval"
    require_non_empty_comment: bool = True
    stratify_by: str = "comment_source"


def _hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_stratum(row: dict[str, Any], stratify_by: str) -> str:
    value = row.get(stratify_by)
    if value in (None, ""):
        return "unknown"
    return str(value)


def _allocate_quotas(
    counts_by_stratum: dict[str, int],
    limit: int,
) -> dict[str, int]:
    if limit <= 0 or not counts_by_stratum:
        return {stratum: 0 for stratum in counts_by_stratum}
    total = sum(counts_by_stratum.values())
    if total <= limit:
        return dict(counts_by_stratum)

    quotas = {stratum: 0 for stratum in counts_by_stratum}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for stratum, count in counts_by_stratum.items():
        ideal = (limit * count) / total
        base = min(count, int(ideal))
        quotas[stratum] = base
        assigned += base
        remainders.append((ideal - base, stratum))

    seats_left = limit - assigned
    for _remainder, stratum in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if seats_left <= 0:
            break
        if quotas[stratum] >= counts_by_stratum[stratum]:
            continue
        quotas[stratum] += 1
        seats_left -= 1

    if seats_left > 0:
        for stratum in sorted(counts_by_stratum):
            if seats_left <= 0:
                break
            available = counts_by_stratum[stratum] - quotas[stratum]
            if available <= 0:
                continue
            take = min(available, seats_left)
            quotas[stratum] += take
            seats_left -= take
    return quotas


def build_comment_retrieval_eval_regime(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_v1",
    output_root: str | Path = "outputs/week10/comment_retrieval",
    config: CommentRetrievalEvalConfig | None = None,
) -> dict[str, Any]:
    """Build deterministic evaluation subsets and a manifest for comment retrieval."""
    eval_config = config or CommentRetrievalEvalConfig()
    input_dir = Path(input_root)
    output_dir = Path(output_root)
    subset_dir = output_dir / "probe_subset"
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)

    limits = {
        "train": eval_config.train_limit,
        "val": eval_config.val_limit,
        "test": eval_config.test_limit,
    }
    manifest: dict[str, Any] = {
        "input_root": str(input_dir),
        "evaluation_mode": "fixed_stratified_subset",
        "sampling_rule": "stable-hash-ranked within split and comment_source strata",
        "source_proportion_rule": "single-source corpus; source proportions are preserved trivially",
        "config": asdict(eval_config),
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        rows = [dict(row) for row in iter_records_from_path(input_dir / f"{split_name}.jsonl")]
        eligible_rows = [
            row
            for row in rows
            if not eval_config.require_non_empty_comment
            or bool(str(row.get("comment_text") or "").strip())
        ]
        stratum_rows: dict[str, list[dict[str, Any]]] = {}
        for row in eligible_rows:
            stratum = _row_stratum(row, eval_config.stratify_by)
            stratum_rows.setdefault(stratum, []).append(row)
        full_counts = {stratum: len(stratum_rows[stratum]) for stratum in sorted(stratum_rows)}
        quotas = _allocate_quotas(full_counts, limits[split_name])

        selected_rows: list[dict[str, Any]] = []
        subset_counts: Counter[str] = Counter()
        for stratum, payload_rows in sorted(stratum_rows.items()):
            ranked_rows = sorted(
                payload_rows,
                key=lambda row: _hash_key(f"{eval_config.salt}:{split_name}:{row.get('probe_id') or row.get('sidecar_id')}"),
            )
            chosen_rows = ranked_rows[: quotas[stratum]]
            selected_rows.extend(chosen_rows)
            subset_counts[stratum] = len(chosen_rows)
        selected_rows.sort(
            key=lambda row: _hash_key(f"{eval_config.salt}:{split_name}:{row.get('probe_id') or row.get('sidecar_id')}")
        )

        output_path = subset_dir / f"annotated_sidecar_{split_name}.jsonl"
        write_jsonl(output_path, selected_rows)
        manifest["splits"][split_name] = {
            "input_rows": len(rows),
            "eligible_rows": len(eligible_rows),
            "selected_rows": len(selected_rows),
            "stratify_by": eval_config.stratify_by,
            "full_stratum_counts": full_counts,
            "subset_stratum_counts": dict(subset_counts),
            "quotas": quotas,
            "output_path": str(output_path),
        }

    manifest_path = output_dir / "retrieval_eval_manifest.yaml"
    write_yaml(manifest_path, manifest)
    return {
        "manifest_path": str(manifest_path),
        "subset_root": str(subset_dir),
        "manifest": manifest,
    }
