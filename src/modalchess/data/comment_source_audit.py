"""Source-family auditing for move-conditioned comment corpora."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from statistics import fmean
from typing import Any, Mapping
from urllib.parse import urlparse

from modalchess.data.comment_informativeness import (
    CommentInformativenessConfig,
    compute_comment_informativeness,
)
from modalchess.data.preprocessing_common import iter_records_from_path, special_rule_flags


def _slugify_label(value: str) -> str:
    collapsed = re.sub(r"\s+", "_", value.strip().lower())
    collapsed = re.sub(r"[^a-z0-9._:@/-]+", "_", collapsed)
    collapsed = re.sub(r"_+", "_", collapsed).strip("_")
    return collapsed or "unknown"


def _domain_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if "://" in text:
        parsed = urlparse(text)
        domain = parsed.netloc.lower()
    else:
        domain = text.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    if "/" in domain:
        domain = domain.split("/", 1)[0]
    if "." in domain and " " not in domain:
        return domain
    return None


def derive_comment_source_family(row: Mapping[str, Any]) -> str:
    """Derive a stable source-family label from explicit fields or metadata."""
    explicit = row.get("source_family")
    if explicit not in (None, ""):
        return _slugify_label(str(explicit))

    source_name = str(row.get("source") or "unknown")
    if source_name.startswith("mate_dataset_"):
        return _slugify_label(source_name)
    if source_name.startswith("waterhorse_annotated_pgn"):
        metadata = row.get("metadata")
        headers: Mapping[str, Any] = {}
        if isinstance(metadata, Mapping):
            header_payload = metadata.get("headers")
            if isinstance(header_payload, Mapping):
                headers = header_payload
        for candidate in (
            headers.get("Site"),
            headers.get("Annotator"),
            metadata.get("annotator") if isinstance(metadata, Mapping) else None,
        ):
            domain = _domain_or_none(candidate)
            if domain is not None:
                return domain
        site = str(headers.get("Site") or "").strip()
        event = str(headers.get("Event") or "").strip()
        annotator = str(headers.get("Annotator") or "").strip()
        for candidate in (site, event, annotator):
            if candidate and candidate != "?":
                return f"waterhorse::{_slugify_label(candidate)[:64]}"
        return "waterhorse::unknown"

    source_file = str(row.get("source_file") or "")
    if source_file:
        path = Path(source_file)
        if path.stem:
            return _slugify_label(path.stem)
    return _slugify_label(source_name)


@dataclass(slots=True)
class CommentSourceAuditConfig:
    informativeness_config: CommentInformativenessConfig = field(default_factory=CommentInformativenessConfig)
    max_example_rows: int = 5


def _load_rows_by_split(input_root: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = Path(input_root)
    return {
        split_name: [dict(row) for row in iter_records_from_path(root / f"{split_name}.jsonl")]
        for split_name in ("train", "val", "test")
    }


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }
    ordered = sorted(values)

    def _pct(percentile: float) -> float:
        index = int(round((len(ordered) - 1) * percentile))
        return float(ordered[index])

    return {
        "mean": float(fmean(ordered)),
        "p10": _pct(0.10),
        "p50": _pct(0.50),
        "p90": _pct(0.90),
    }


def generate_comment_source_family_audit(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_v3_informative",
    config: CommentSourceAuditConfig | None = None,
) -> dict[str, Any]:
    audit_config = config or CommentSourceAuditConfig()
    rows_by_split = _load_rows_by_split(input_root)

    rows_by_source_file: Counter[str] = Counter()
    rows_by_comment_source: Counter[str] = Counter()
    rows_by_source_family: Counter[str] = Counter()
    duplicate_rows_by_source_family: Counter[str] = Counter()
    informativeness_by_source_family: dict[str, list[float]] = defaultdict(list)
    special_rules_by_source_family: dict[str, Counter[str]] = defaultdict(Counter)
    examples_by_source_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    normalized_comment_clusters: dict[str, Counter[str]] = defaultdict(Counter)

    for split_name, rows in rows_by_split.items():
        for row in rows:
            source_file = str(row.get("source_file") or "<none>")
            comment_source = str(row.get("comment_source") or "unknown")
            source_family = derive_comment_source_family(row)
            rows_by_source_file[source_file] += 1
            rows_by_comment_source[comment_source] += 1
            rows_by_source_family[source_family] += 1
            family_split_counts[source_family][split_name] += 1

            info_payload = row.get("comment_informativeness")
            if isinstance(info_payload, Mapping) and info_payload.get("informativeness_score") is not None:
                score = float(info_payload["informativeness_score"])
            elif row.get("informativeness_score") is not None:
                score = float(row["informativeness_score"])
            else:
                score = float(
                    compute_comment_informativeness(dict(row), config=audit_config.informativeness_config)[
                        "informativeness_score"
                    ]
                )
            informativeness_by_source_family[source_family].append(score)

            flags = special_rule_flags(str(row["fen"]), str(row["target_move_uci"]))
            for key, value in flags.items():
                special_rules_by_source_family[source_family][key] += int(value)

            comment_text = str(row.get("comment_text") or "").strip()
            if comment_text:
                normalized_comment_clusters[source_family][comment_text] += 1

            example_rows = examples_by_source_family[source_family]
            if len(example_rows) < audit_config.max_example_rows:
                example_rows.append(
                    {
                        "sidecar_id": row.get("sidecar_id"),
                        "split": split_name,
                        "comment_source": comment_source,
                        "comment_text": comment_text[:240],
                    }
                )

    for source_family, counter in normalized_comment_clusters.items():
        duplicate_rows_by_source_family[source_family] = sum(count - 1 for count in counter.values() if count > 1)

    total_rows = sum(rows_by_source_family.values())
    sorted_families = sorted(rows_by_source_family.items(), key=lambda item: (-item[1], item[0]))
    family_summary: dict[str, Any] = {}
    for source_family, count in sorted_families:
        family_summary[source_family] = {
            "row_count": count,
            "row_share": (count / total_rows) if total_rows else 0.0,
            "split_counts": dict(family_split_counts[source_family]),
            "duplicate_row_share": (
                duplicate_rows_by_source_family[source_family] / count if count else 0.0
            ),
            "informativeness": _distribution(informativeness_by_source_family[source_family]),
            "special_rule_counts": dict(special_rules_by_source_family[source_family]),
            "examples": examples_by_source_family[source_family],
        }

    top_families = [
        {"source_family": source_family, "row_count": count}
        for source_family, count in sorted_families[:10]
    ]
    repetitive_families = [
        {
            "source_family": source_family,
            "duplicate_row_share": payload["duplicate_row_share"],
            "row_count": payload["row_count"],
        }
        for source_family, payload in sorted(
            family_summary.items(),
            key=lambda item: (-item[1]["duplicate_row_share"], -item[1]["row_count"], item[0]),
        )[:10]
    ]
    informative_families = [
        {
            "source_family": source_family,
            "mean_informativeness": payload["informativeness"]["mean"],
            "row_count": payload["row_count"],
        }
        for source_family, payload in sorted(
            family_summary.items(),
            key=lambda item: (-item[1]["informativeness"]["mean"], -item[1]["row_count"], item[0]),
        )[:10]
    ]

    largest_family_share = top_families[0]["row_count"] / total_rows if top_families and total_rows else 0.0
    report = {
        "input_root": str(input_root),
        "config": asdict(audit_config.informativeness_config),
        "rows_by_source_file": dict(rows_by_source_file.most_common()),
        "rows_by_comment_source": dict(rows_by_comment_source.most_common()),
        "rows_by_source_family": dict(rows_by_source_family.most_common()),
        "source_family_summary": family_summary,
        "top_source_families": top_families,
        "most_repetitive_source_families": repetitive_families,
        "highest_informativeness_source_families": informative_families,
        "source_family_diversity": {
            "family_count": len(rows_by_source_family),
            "families_with_at_least_100_rows": sum(int(count >= 100) for count in rows_by_source_family.values()),
            "largest_family_share": largest_family_share,
            "single_family_dominant": largest_family_share >= 0.5,
        },
    }
    return report


def write_comment_source_family_audit(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_v3_informative",
    output_dir: str | Path | None = None,
    config: CommentSourceAuditConfig | None = None,
) -> dict[str, Any]:
    report = generate_comment_source_family_audit(input_root=input_root, config=config)
    report_dir = Path(output_dir) if output_dir is not None else Path(input_root) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "source_family_audit.json"
    md_path = report_dir / "source_family_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Source Family Audit", ""]
    lines.append("## Top Source Families")
    for item in report["top_source_families"]:
        lines.append(f"- `{item['source_family']}`: {item['row_count']}")
    lines.append("")
    lines.append("## Most Repetitive Source Families")
    for item in report["most_repetitive_source_families"]:
        lines.append(
            f"- `{item['source_family']}`: duplicate_row_share={item['duplicate_row_share']:.4f}, rows={item['row_count']}"
        )
    lines.append("")
    lines.append("## Highest Informativeness Source Families")
    for item in report["highest_informativeness_source_families"]:
        lines.append(
            f"- `{item['source_family']}`: mean_informativeness={item['mean_informativeness']:.4f}, rows={item['row_count']}"
        )
    lines.append("")
    diversity = report["source_family_diversity"]
    lines.append("## Diversity")
    lines.append(f"- family_count: {diversity['family_count']}")
    lines.append(f"- families_with_at_least_100_rows: {diversity['families_with_at_least_100_rows']}")
    lines.append(f"- largest_family_share: {diversity['largest_family_share']:.4f}")
    lines.append(f"- single_family_dominant: {diversity['single_family_dominant']}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "report": report,
    }
