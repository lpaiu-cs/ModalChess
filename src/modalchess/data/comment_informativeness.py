"""Score comment informativeness for cleaned annotated comment corpora."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import statistics
from typing import Any

from modalchess.data.comment_boilerplate_audit import MARKUP_RE
from modalchess.data.preprocessing_common import iter_records_from_path


WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[a-z0-9']+")
SQUARE_RE = re.compile(r"\b[a-h][1-8]\b", re.IGNORECASE)
RESULT_TEXT_RE = re.compile(r"^(1-0|0-1|1/2-1/2)(\s+.+)?\.?$", re.IGNORECASE)

CHESS_WORDS = {
    "attack",
    "attacks",
    "back",
    "bishop",
    "capture",
    "captures",
    "castle",
    "castles",
    "castling",
    "center",
    "central",
    "check",
    "checkmate",
    "counterplay",
    "defend",
    "defends",
    "diagonal",
    "endgame",
    "exchange",
    "file",
    "fork",
    "king",
    "knight",
    "mate",
    "pawn",
    "pin",
    "promotion",
    "queen",
    "rank",
    "recapture",
    "rook",
    "skewer",
    "square",
    "tactic",
    "threat",
}

EXPLANATORY_WORDS = {
    "because",
    "but",
    "however",
    "if",
    "since",
    "so",
    "therefore",
    "thus",
    "while",
    "with",
}

MOVE_ANCHOR_WORDS = {
    "capture",
    "captures",
    "castle",
    "castles",
    "castling",
    "check",
    "checkmate",
    "mate",
    "promotion",
    "promotes",
    "recapture",
    "takes",
    "threat",
    "threatens",
}


@dataclass(slots=True)
class CommentInformativenessConfig:
    low_threshold: float = 0.33
    medium_threshold: float = 0.45
    high_threshold: float = 0.62
    medium_source_quantile: float = 0.60
    special_rule_floor: float = 0.30


def _load_rows_by_split(root: str | Path) -> dict[str, list[dict[str, Any]]]:
    input_root = Path(root)
    return {
        split_name: [dict(row) for row in iter_records_from_path(input_root / f"{split_name}.jsonl")]
        for split_name in ("train", "val", "test")
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = int(round((len(sorted_values) - 1) * percentile))
    return float(sorted_values[rank])


def _feature_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "p10": _percentile(values, 0.10),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": float(max(values)),
    }


def _word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def compute_comment_informativeness(
    row: dict[str, Any],
    *,
    config: CommentInformativenessConfig | None = None,
) -> dict[str, Any]:
    score_config = config or CommentInformativenessConfig()
    text = str(row.get("comment_text") or "").strip()
    original_text = str(row.get("original_comment_text") or text).strip()
    tokens = _word_tokens(text)
    alpha_tokens = [token for token in tokens if any(character.isalpha() for character in token)]
    numeric_tokens = [token for token in tokens if token.isdigit()]
    lexical_diversity = (len(set(tokens)) / len(tokens)) if tokens else 0.0
    markup_matches = list(MARKUP_RE.finditer(original_text))
    markup_ratio = (sum(len(match.group(0)) for match in markup_matches) / len(original_text)) if original_text else 0.0
    symbol_char_count = sum(
        1 for character in text if not character.isalnum() and not character.isspace()
    )
    symbol_ratio = (symbol_char_count / len(text)) if text else 0.0
    chess_word_hits = sum(1 for token in tokens if token in CHESS_WORDS)
    explanatory_hits = sum(1 for token in tokens if token in EXPLANATORY_WORDS)
    move_anchor_hits = sum(1 for token in tokens if token in MOVE_ANCHOR_WORDS)
    move_anchor_hits += len(SQUARE_RE.findall(text))
    alpha_token_ratio = (len(alpha_tokens) / len(tokens)) if tokens else 0.0
    numeric_token_ratio = (len(numeric_tokens) / len(tokens)) if tokens else 0.0
    mostly_numeric_or_result = bool(
        RESULT_TEXT_RE.match(text)
        or (tokens and numeric_token_ratio >= 0.5)
        or markup_ratio >= 0.5
    )

    length_score = _clamp((min(len(text), 220) - 20.0) / 120.0)
    token_score = _clamp((min(len(tokens), 32) - 4.0) / 12.0)
    lexical_score = _clamp((lexical_diversity - 0.35) / 0.45)
    alpha_score = _clamp(alpha_token_ratio)
    chess_anchor_score = _clamp(chess_word_hits / 2.0)
    explanatory_score = 1.0 if explanatory_hits > 0 else 0.0
    move_anchor_score = 1.0 if move_anchor_hits > 0 else 0.0

    score = (
        0.18 * length_score
        + 0.16 * token_score
        + 0.16 * lexical_score
        + 0.10 * alpha_score
        + 0.16 * chess_anchor_score
        + 0.12 * explanatory_score
        + 0.12 * move_anchor_score
    )
    penalty = 0.0
    if mostly_numeric_or_result:
        penalty += 0.35
    if symbol_ratio >= 0.25:
        penalty += 0.10
    if alpha_token_ratio < 0.5:
        penalty += 0.08
    score = _clamp(score - penalty)

    if score >= score_config.high_threshold:
        bucket = "high"
    elif score >= score_config.medium_threshold:
        bucket = "medium"
    else:
        bucket = "low"

    return {
        "char_length": len(text),
        "token_count": len(tokens),
        "lexical_diversity": lexical_diversity,
        "alpha_token_ratio": alpha_token_ratio,
        "markup_ratio": markup_ratio,
        "symbol_ratio": symbol_ratio,
        "chess_word_hits": chess_word_hits,
        "explanatory_hits": explanatory_hits,
        "move_anchor_hits": move_anchor_hits,
        "mostly_numeric_or_result": mostly_numeric_or_result,
        "informativeness_score": score,
        "informativeness_bucket": bucket,
    }


def annotate_informativeness_rows(
    *,
    input_root: str | Path,
    config: CommentInformativenessConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    score_config = config or CommentInformativenessConfig()
    rows_by_split = _load_rows_by_split(input_root)
    for rows in rows_by_split.values():
        for row in rows:
            row["comment_informativeness"] = compute_comment_informativeness(row, config=score_config)
    return rows_by_split


def generate_comment_informativeness_audit(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_v2_clean",
    config: CommentInformativenessConfig | None = None,
) -> dict[str, Any]:
    score_config = config or CommentInformativenessConfig()
    rows_by_split = annotate_informativeness_rows(input_root=input_root, config=score_config)
    all_rows = [row for rows in rows_by_split.values() for row in rows]

    score_distribution_by_split = {
        split_name: _feature_distribution(
            [float(row["comment_informativeness"]["informativeness_score"]) for row in rows]
        )
        for split_name, rows in rows_by_split.items()
    }
    score_distribution_by_source: dict[str, list[float]] = defaultdict(list)
    bucket_counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    bucket_counts_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    low_template_counter: Counter[str] = Counter()
    high_examples: list[dict[str, Any]] = []
    duplicate_counter: Counter[str] = Counter()

    for row in all_rows:
        score_payload = row["comment_informativeness"]
        source_name = str(row.get("comment_source") or "unknown")
        bucket = str(score_payload["informativeness_bucket"])
        score_distribution_by_source[source_name].append(float(score_payload["informativeness_score"]))
        bucket_counts_by_split[str(row.get("split") or "unknown")][bucket] += 1
        bucket_counts_by_source[source_name][bucket] += 1
        template = str(row.get("comment_template") or row.get("comment_text") or "").strip()
        if template:
            duplicate_counter[template] += 1
        if bucket == "low" and template:
            low_template_counter[template] += 1
        if bucket == "high" and len(high_examples) < 25:
            high_examples.append(
                {
                    "sidecar_id": row.get("sidecar_id"),
                    "split": row.get("split"),
                    "comment_source": row.get("comment_source"),
                    "comment_text": row.get("comment_text"),
                    "informativeness_score": score_payload["informativeness_score"],
                }
            )

    duplicate_correlation: dict[str, dict[str, float]] = {}
    for bucket_name in ("low", "medium", "high"):
        bucket_rows = [row for row in all_rows if row["comment_informativeness"]["informativeness_bucket"] == bucket_name]
        if not bucket_rows:
            duplicate_correlation[bucket_name] = {"mean_cluster_size": 0.0, "duplicate_row_share": 0.0}
            continue
        cluster_sizes = [
            duplicate_counter.get(str(row.get("comment_template") or row.get("comment_text") or "").strip(), 1)
            for row in bucket_rows
        ]
        duplicate_correlation[bucket_name] = {
            "mean_cluster_size": float(statistics.fmean(cluster_sizes)),
            "duplicate_row_share": (
                sum(int(cluster_size > 1) for cluster_size in cluster_sizes) / len(cluster_sizes)
            ),
        }

    report = {
        "input_root": str(input_root),
        "config": asdict(score_config),
        "total_rows": len(all_rows),
        "score_distribution_by_split": score_distribution_by_split,
        "score_distribution_by_comment_source": {
            source_name: _feature_distribution(values)
            for source_name, values in sorted(score_distribution_by_source.items())
        },
        "bucket_counts_by_split": {split_name: dict(counter) for split_name, counter in bucket_counts_by_split.items()},
        "bucket_counts_by_comment_source": {source_name: dict(counter) for source_name, counter in bucket_counts_by_source.items()},
        "top_low_information_templates": [
            {"comment_text": template, "count": count}
            for template, count in low_template_counter.most_common(20)
        ],
        "high_information_examples": high_examples[:10],
        "duplicate_cluster_correlation_by_bucket": duplicate_correlation,
    }
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Comment Informativeness Audit", ""]
    lines.append(f"- total_rows: {report['total_rows']}")
    lines.append("")
    lines.append("## Score Distribution By Split")
    for split_name, stats in report["score_distribution_by_split"].items():
        lines.append(
            f"- `{split_name}`: mean={stats['mean']:.4f}, p10={stats['p10']:.4f}, "
            f"p50={stats['p50']:.4f}, p90={stats['p90']:.4f}, p95={stats['p95']:.4f}"
        )
    lines.append("")
    lines.append("## Bucket Counts By Split")
    for split_name, counts in report["bucket_counts_by_split"].items():
        lines.append(f"- `{split_name}`: {counts}")
    lines.append("")
    lines.append("## Top Low-Information Templates")
    for row in report["top_low_information_templates"][:10]:
        lines.append(f"- `{row['comment_text']}`: {row['count']}")
    lines.append("")
    lines.append("## High-Information Examples")
    for row in report["high_information_examples"][:5]:
        lines.append(
            f"- score={row['informativeness_score']:.4f}, split={row['split']}, "
            f"source={row['comment_source']}: `{row['comment_text']}`"
        )
    return "\n".join(lines) + "\n"


def write_comment_informativeness_audit(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_v2_clean",
    output_dir: str | Path | None = None,
    config: CommentInformativenessConfig | None = None,
) -> dict[str, str]:
    input_path = Path(input_root)
    report_root = Path(output_dir) if output_dir is not None else input_path / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    report = generate_comment_informativeness_audit(input_root=input_path, config=config)
    json_path = report_root / "comment_informativeness_audit.json"
    md_path = report_root / "comment_informativeness_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return {"report_json": str(json_path), "report_md": str(md_path)}
