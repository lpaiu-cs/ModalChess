"""Audit style divergence across annotated comment sources and source families."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import statistics
from typing import Any

from modalchess.data.comment_boilerplate_audit import (
    CommentBoilerplateConfig,
    analyze_comment_text,
    normalized_template_text,
)
from modalchess.data.comment_informativeness import CommentInformativenessConfig, compute_comment_informativeness
from modalchess.data.preprocessing_common import iter_records_from_path, special_rule_flags


@dataclass(slots=True)
class CommentSourceStyleAuditConfig:
    boilerplate_config: CommentBoilerplateConfig = field(default_factory=CommentBoilerplateConfig)
    informativeness_config: CommentInformativenessConfig = field(default_factory=CommentInformativenessConfig)
    min_group_rows: int = 100


def _load_rows(root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_root = Path(root)
    for split_name in ("train", "val", "test"):
        split_path = input_root / f"{split_name}.jsonl"
        if not split_path.exists():
            continue
        for row in iter_records_from_path(split_path):
            payload = dict(row)
            payload.setdefault("split", split_name)
            rows.append(payload)
    return rows


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = int(round((len(sorted_values) - 1) * percentile))
    return float(sorted_values[rank])


def _score_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
    }


def _style_cluster(metrics: dict[str, Any]) -> str:
    markup_heavy_rate = float(metrics["markup_heavy_rate"])
    numeric_symbol_rate = float(metrics["numeric_symbol_heavy_rate"])
    duplicate_rate = float(metrics["duplicate_rate"])
    informativeness_mean = float(metrics["informativeness_distribution"]["mean"])
    if markup_heavy_rate >= 0.20 or numeric_symbol_rate >= 0.20:
        return "markup_heavy"
    if duplicate_rate >= 0.25:
        return "repetitive"
    if informativeness_mean >= 0.68 and markup_heavy_rate < 0.10:
        return "human_leaning"
    return "mixed_style"


def _summarize_group(
    group_name: str,
    rows: list[dict[str, Any]],
    *,
    template_counter: Counter[str],
    config: CommentSourceStyleAuditConfig,
) -> dict[str, Any]:
    char_lengths: list[float] = []
    token_counts: list[float] = []
    lexical_diversities: list[float] = []
    markup_shares: list[float] = []
    informativeness_scores: list[float] = []
    numeric_symbol_heavy = 0
    markup_heavy = 0
    duplicate_rows = 0
    comment_source_counts: Counter[str] = Counter()
    special_rule_counts: Counter[str] = Counter()

    for row in rows:
        text = str(row.get("comment_text") or "")
        boilerplate = analyze_comment_text(
            text,
            template_count=template_counter.get(normalized_template_text(text), 0),
            config=config.boilerplate_config,
        )
        info = compute_comment_informativeness(row, config=config.informativeness_config)
        char_lengths.append(float(info["char_length"]))
        token_counts.append(float(info["token_count"]))
        lexical_diversities.append(float(info["lexical_diversity"]))
        markup_shares.append(float(boilerplate["markup_char_share"]))
        informativeness_scores.append(float(info["informativeness_score"]))
        markup_heavy += int("pgn_markup_heavy" in boilerplate["categories"] or "markup_only" in boilerplate["categories"])
        numeric_symbol_heavy += int(info["mostly_numeric_or_result"] or info["symbol_ratio"] >= 0.25)
        duplicate_rows += int(template_counter.get(boilerplate["normalized_template"], 0) > 1 and bool(boilerplate["normalized_template"]))
        comment_source_counts[str(row.get("comment_source") or "unknown")] += 1
        flags = special_rule_flags(str(row["fen"]), str(row["target_move_uci"]) if row.get("target_move_uci") else None)
        for key, value in flags.items():
            special_rule_counts[key] += int(value)

    row_count = len(rows)
    summary = {
        "group_name": group_name,
        "row_count": row_count,
        "avg_char_length": float(statistics.fmean(char_lengths)) if char_lengths else 0.0,
        "avg_token_count": float(statistics.fmean(token_counts)) if token_counts else 0.0,
        "avg_lexical_diversity": float(statistics.fmean(lexical_diversities)) if lexical_diversities else 0.0,
        "avg_markup_rate": float(statistics.fmean(markup_shares)) if markup_shares else 0.0,
        "markup_heavy_rate": (markup_heavy / row_count) if row_count else 0.0,
        "numeric_symbol_heavy_rate": (numeric_symbol_heavy / row_count) if row_count else 0.0,
        "duplicate_rate": (duplicate_rows / row_count) if row_count else 0.0,
        "informativeness_distribution": _score_distribution(informativeness_scores),
        "comment_source_composition": {
            key: (count / row_count) if row_count else 0.0
            for key, count in sorted(comment_source_counts.items())
        },
        "special_rule_coverage": dict(sorted(special_rule_counts.items())),
    }
    summary["style_cluster"] = _style_cluster(summary)
    return summary


def _divergence_scores(group_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not group_rows:
        return []
    feature_names = (
        "avg_char_length",
        "avg_token_count",
        "avg_lexical_diversity",
        "avg_markup_rate",
        "duplicate_rate",
        "numeric_symbol_heavy_rate",
    )
    means = {
        feature_name: statistics.fmean(float(row[feature_name]) for row in group_rows)
        for feature_name in feature_names
    }
    stds = {}
    for feature_name in feature_names:
        values = [float(row[feature_name]) for row in group_rows]
        stds[feature_name] = statistics.stdev(values) if len(values) > 1 else 1.0
        if stds[feature_name] < 1e-8:
            stds[feature_name] = 1.0
    scored_rows: list[dict[str, Any]] = []
    for row in group_rows:
        distance = math.sqrt(
            sum(
                ((float(row[feature_name]) - means[feature_name]) / stds[feature_name]) ** 2
                for feature_name in feature_names
            )
        )
        scored_rows.append({**row, "style_divergence_score": distance})
    return sorted(scored_rows, key=lambda row: (-float(row["style_divergence_score"]), -int(row["row_count"]), row["group_name"]))


def generate_comment_source_style_audit(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_v4_multisource",
    config: CommentSourceStyleAuditConfig | None = None,
) -> dict[str, Any]:
    audit_config = config or CommentSourceStyleAuditConfig()
    rows = _load_rows(input_root)
    template_counter: Counter[str] = Counter(
        template
        for template in (normalized_template_text(row.get("comment_text")) for row in rows)
        if template
    )

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source") or "unknown")].append(row)
        by_family[str(row.get("source_family") or "unknown")].append(row)

    source_metrics = [
        _summarize_group(source_name, group_rows, template_counter=template_counter, config=audit_config)
        for source_name, group_rows in sorted(by_source.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    family_metrics = [
        _summarize_group(family_name, group_rows, template_counter=template_counter, config=audit_config)
        for family_name, group_rows in sorted(by_family.items(), key=lambda item: (-len(item[1]), item[0]))
        if len(group_rows) >= audit_config.min_group_rows
    ]
    divergent_families = _divergence_scores(family_metrics)
    markup_heavy = sorted(family_metrics, key=lambda row: (-float(row["markup_heavy_rate"]), -int(row["row_count"]), row["group_name"]))
    informative = sorted(
        family_metrics,
        key=lambda row: (-float(row["informativeness_distribution"]["mean"]), -int(row["row_count"]), row["group_name"]),
    )
    cluster_summary = dict(
        Counter(str(row["style_cluster"]) for row in family_metrics).most_common()
    )

    return {
        "input_root": str(input_root),
        "config": asdict(audit_config),
        "row_count": len(rows),
        "rows_by_source": source_metrics,
        "rows_by_source_family": family_metrics,
        "most_style_divergent_families": divergent_families[:10],
        "most_markup_heavy_families": markup_heavy[:10],
        "most_informative_families": informative[:10],
        "source_style_cluster_summary": cluster_summary,
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Source Style Audit", ""]
    lines.append(f"- row_count: {report['row_count']}")
    lines.append(f"- source_style_cluster_summary: {report['source_style_cluster_summary']}")
    lines.append("")
    lines.append("## Most Style-Divergent Families")
    for row in report["most_style_divergent_families"][:10]:
        lines.append(
            f"- `{row['group_name']}`: divergence={row['style_divergence_score']:.4f}, "
            f"rows={row['row_count']}, markup_heavy={row['markup_heavy_rate']:.4f}, "
            f"informativeness_mean={row['informativeness_distribution']['mean']:.4f}"
        )
    lines.append("")
    lines.append("## Most Markup-Heavy Families")
    for row in report["most_markup_heavy_families"][:10]:
        lines.append(
            f"- `{row['group_name']}`: markup_heavy={row['markup_heavy_rate']:.4f}, "
            f"avg_markup_rate={row['avg_markup_rate']:.4f}, rows={row['row_count']}"
        )
    lines.append("")
    lines.append("## Most Informative Families")
    for row in report["most_informative_families"][:10]:
        lines.append(
            f"- `{row['group_name']}`: informativeness_mean={row['informativeness_distribution']['mean']:.4f}, "
            f"duplicate_rate={row['duplicate_rate']:.4f}, rows={row['row_count']}"
        )
    return "\n".join(lines) + "\n"


def write_comment_source_style_audit(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_v4_multisource",
    output_dir: str | Path | None = None,
    config: CommentSourceStyleAuditConfig | None = None,
) -> dict[str, str]:
    input_path = Path(input_root)
    report_root = Path(output_dir) if output_dir is not None else input_path / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    report = generate_comment_source_style_audit(input_root=input_path, config=config)
    json_path = report_root / "source_style_audit.json"
    md_path = report_root / "source_style_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return {"report_json": str(json_path), "report_md": str(md_path)}
