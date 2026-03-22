"""Duplicate and normalization audit for annotated comment sidecars."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
from typing import Any

from modalchess.data.preprocessing_common import iter_records_from_path


WHITESPACE_RE = re.compile(r"\s+")
PUNCT_LIGHT_RE = re.compile(r"[^\w\s]")
NAG_TOKEN_RE = re.compile(r"\$\d+")


def normalize_comment_text(text: str | None, *, mode: str = "raw") -> str:
    raw = str(text or "").strip()
    if mode == "raw":
        return raw
    if mode == "lower_ws":
        return WHITESPACE_RE.sub(" ", raw.lower()).strip()
    if mode == "punct_light":
        lowered = WHITESPACE_RE.sub(" ", raw.lower()).strip()
        punct_light = PUNCT_LIGHT_RE.sub(" ", lowered)
        return WHITESPACE_RE.sub(" ", punct_light).strip()
    if mode == "nag_stripped":
        lowered = WHITESPACE_RE.sub(" ", raw.lower()).strip()
        nag_stripped = NAG_TOKEN_RE.sub(" ", lowered)
        punct_light = PUNCT_LIGHT_RE.sub(" ", nag_stripped)
        return WHITESPACE_RE.sub(" ", punct_light).strip()
    raise ValueError(f"unsupported comment normalization mode: {mode}")


def _load_rows_by_split(root: str | Path) -> dict[str, list[dict[str, Any]]]:
    input_root = Path(root)
    return {
        split_name: [dict(row) for row in iter_records_from_path(input_root / f"{split_name}.jsonl")]
        for split_name in ("train", "val", "test")
    }


def _length_stats(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
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


def _top_repeated_comment_strings(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    sources_by_text: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        comment = normalize_comment_text(row.get("comment_text"), mode=mode)
        if not comment:
            continue
        counter[comment] += 1
        sources_by_text[comment][str(row.get("comment_source") or "unknown")] += 1
    result: list[dict[str, Any]] = []
    for comment, count in counter.most_common(limit):
        result.append(
            {
                "comment_text": comment,
                "count": count,
                "comment_source_counts": dict(sources_by_text[comment]),
            }
        )
    return result


def _cluster_rows_by_key(
    rows: list[dict[str, Any]],
    *,
    key_fn,
) -> dict[str, list[dict[str, Any]]]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(key_fn(row))
        if not key:
            continue
        clusters[key].append(row)
    return clusters


def _cluster_summary(
    clusters: dict[str, list[dict[str, Any]]],
    *,
    row_count: int,
) -> dict[str, Any]:
    duplicate_clusters = [rows for rows in clusters.values() if len(rows) > 1]
    duplicate_rows = sum(len(rows) - 1 for rows in duplicate_clusters)
    cluster_sizes = sorted((len(rows) for rows in duplicate_clusters), reverse=True)
    lengths = [len(str(row.get("comment_text") or "")) for rows in duplicate_clusters for row in rows]
    return {
        "cluster_count": len(duplicate_clusters),
        "duplicate_row_count": duplicate_rows,
        "duplicate_row_share": (duplicate_rows / row_count) if row_count else 0.0,
        "largest_cluster_sizes": cluster_sizes[:20],
        "duplicate_comment_length_chars": _length_stats(lengths),
    }


def _duplicate_share_by_split(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    key_fn,
) -> dict[str, float]:
    shares: dict[str, float] = {}
    for split_name, rows in rows_by_split.items():
        clusters = _cluster_rows_by_key(rows, key_fn=key_fn)
        duplicate_rows = sum(len(cluster_rows) - 1 for cluster_rows in clusters.values() if len(cluster_rows) > 1)
        shares[split_name] = (duplicate_rows / len(rows)) if rows else 0.0
    return shares


def _duplicate_share_by_comment_source(
    rows: list[dict[str, Any]],
    *,
    key_fn,
) -> dict[str, float]:
    rows_by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_source[str(row.get("comment_source") or "unknown")].append(row)
    shares: dict[str, float] = {}
    for source_name, source_rows in rows_by_source.items():
        clusters = _cluster_rows_by_key(source_rows, key_fn=key_fn)
        duplicate_rows = sum(len(cluster_rows) - 1 for cluster_rows in clusters.values() if len(cluster_rows) > 1)
        shares[source_name] = (duplicate_rows / len(source_rows)) if source_rows else 0.0
    return shares


def generate_comment_duplicate_audit(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_v1",
) -> dict[str, Any]:
    rows_by_split = _load_rows_by_split(input_root)
    all_rows = [row for rows in rows_by_split.values() for row in rows]

    exact_comment_clusters = _cluster_rows_by_key(
        all_rows,
        key_fn=lambda row: normalize_comment_text(row.get("comment_text"), mode="raw"),
    )
    exact_fen_move_comment_clusters = _cluster_rows_by_key(
        all_rows,
        key_fn=lambda row: json.dumps(
            [
                str(row.get("fen") or ""),
                str(row.get("target_move_uci") or ""),
                normalize_comment_text(row.get("comment_text"), mode="raw"),
            ],
            ensure_ascii=False,
        ),
    )
    cross_position_clusters: dict[str, list[dict[str, Any]]] = {}
    for comment_text, comment_rows in exact_comment_clusters.items():
        if len(comment_rows) <= 1 or not comment_text:
            continue
        unique_positions = {
            (str(row.get("fen") or ""), str(row.get("target_move_uci") or ""))
            for row in comment_rows
        }
        if len(unique_positions) > 1:
            cross_position_clusters[comment_text] = comment_rows

    normalization_views = {}
    for mode in ("raw", "lower_ws", "punct_light", "nag_stripped"):
        clusters = _cluster_rows_by_key(
            all_rows,
            key_fn=lambda row, view_mode=mode: normalize_comment_text(row.get("comment_text"), mode=view_mode),
        )
        normalization_views[mode] = {
            "cluster_summary": _cluster_summary(clusters, row_count=len(all_rows)),
            "duplicate_share_by_split": _duplicate_share_by_split(
                rows_by_split,
                key_fn=lambda row, view_mode=mode: normalize_comment_text(row.get("comment_text"), mode=view_mode),
            ),
            "duplicate_share_by_comment_source": _duplicate_share_by_comment_source(
                all_rows,
                key_fn=lambda row, view_mode=mode: normalize_comment_text(row.get("comment_text"), mode=view_mode),
            ),
        }

    report = {
        "input_root": str(input_root),
        "total_rows": len(all_rows),
        "rows_by_split": {split_name: len(rows) for split_name, rows in rows_by_split.items()},
        "exact_comment_text": {
            **_cluster_summary(exact_comment_clusters, row_count=len(all_rows)),
            "duplicate_share_by_split": _duplicate_share_by_split(
                rows_by_split,
                key_fn=lambda row: normalize_comment_text(row.get("comment_text"), mode="raw"),
            ),
            "duplicate_share_by_comment_source": _duplicate_share_by_comment_source(
                all_rows,
                key_fn=lambda row: normalize_comment_text(row.get("comment_text"), mode="raw"),
            ),
        },
        "exact_fen_move_comment": _cluster_summary(exact_fen_move_comment_clusters, row_count=len(all_rows)),
        "duplicate_comment_text_across_positions": _cluster_summary(
            cross_position_clusters,
            row_count=len(all_rows),
        ),
        "normalization_views": normalization_views,
        "top_repeated_comment_strings": {
            "raw": _top_repeated_comment_strings(all_rows, mode="raw"),
            "nag_stripped": _top_repeated_comment_strings(all_rows, mode="nag_stripped"),
        },
    }
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Comment Duplicate Audit", ""]
    lines.append(f"- total_rows: {report['total_rows']}")
    lines.append(
        "- rows_by_split: "
        + ", ".join(f"{split_name}={count}" for split_name, count in report["rows_by_split"].items())
    )
    lines.append("")
    lines.append("## Exact Comment Text")
    lines.append(
        f"- duplicate_row_share: {report['exact_comment_text']['duplicate_row_share']:.6f} "
        f"({report['exact_comment_text']['duplicate_row_count']} rows)"
    )
    lines.append(f"- cluster_count: {report['exact_comment_text']['cluster_count']}")
    lines.append(
        f"- largest_cluster_sizes: {report['exact_comment_text']['largest_cluster_sizes'][:10]}"
    )
    lines.append("")
    lines.append("## Exact (FEN, Move, Comment)")
    lines.append(
        f"- duplicate_row_share: {report['exact_fen_move_comment']['duplicate_row_share']:.6f} "
        f"({report['exact_fen_move_comment']['duplicate_row_count']} rows)"
    )
    lines.append("")
    lines.append("## Cross-Position Comment Reuse")
    lines.append(
        f"- duplicate_row_share: {report['duplicate_comment_text_across_positions']['duplicate_row_share']:.6f} "
        f"({report['duplicate_comment_text_across_positions']['duplicate_row_count']} rows)"
    )
    lines.append("")
    lines.append("## Normalization Views")
    for mode, payload in report["normalization_views"].items():
        lines.append(
            f"- `{mode}`: duplicate_row_share={payload['cluster_summary']['duplicate_row_share']:.6f}, "
            f"cluster_count={payload['cluster_summary']['cluster_count']}"
        )
    lines.append("")
    lines.append("## Top Repeated Comments (Raw)")
    for row in report["top_repeated_comment_strings"]["raw"][:10]:
        lines.append(f"- `{row['comment_text']}`: {row['count']}")
    return "\n".join(lines) + "\n"


def write_comment_duplicate_audit(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_v1",
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    input_path = Path(input_root)
    report_root = Path(output_dir) if output_dir is not None else input_path / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    report = generate_comment_duplicate_audit(input_root=input_path)
    json_path = report_root / "comment_duplicate_audit.json"
    md_path = report_root / "comment_duplicate_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return {"report_json": str(json_path), "report_md": str(md_path)}
