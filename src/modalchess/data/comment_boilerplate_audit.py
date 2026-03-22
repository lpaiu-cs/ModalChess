"""Audit and classify low-information boilerplate comments in annotated PGN sidecars."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

from modalchess.data.comment_duplicate_audit import normalize_comment_text
from modalchess.data.preprocessing_common import iter_records_from_path


MARKUP_RE = re.compile(r"\[%([a-zA-Z]+)[^\]]*\]")
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[a-z0-9']+")
RESULT_COMMENT_RE = re.compile(r"^(1-0|0-1|1/2-1/2)(\s+.+)?\.?$", re.IGNORECASE)
SYMBOL_ONLY_RE = re.compile(r"^[!?\.]+$")
ENGINE_TEMPLATE_RE = re.compile(
    r"^(inaccuracy|mistake|blunder)\.\s+.+\s+was best\.?$|^forced\.?$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class CommentBoilerplateConfig:
    repeated_template_min_count: int = 20
    short_template_char_limit: int = 24
    short_template_token_limit: int = 4
    low_lexical_diversity_max: float = 0.50
    low_lexical_token_limit: int = 6
    markup_heavy_share_threshold: float = 0.35


def _load_rows_by_split(root: str | Path) -> dict[str, list[dict[str, Any]]]:
    input_root = Path(root)
    return {
        split_name: [dict(row) for row in iter_records_from_path(input_root / f"{split_name}.jsonl")]
        for split_name in ("train", "val", "test")
    }


def strip_pgn_markup(text: str | None) -> str:
    raw = str(text or "").strip()
    stripped = MARKUP_RE.sub(" ", raw)
    return WHITESPACE_RE.sub(" ", stripped).strip()


def normalized_template_text(text: str | None) -> str:
    return normalize_comment_text(strip_pgn_markup(text), mode="punct_light")


def _markup_tags(text: str) -> list[str]:
    return [match.group(1).lower() for match in MARKUP_RE.finditer(text)]


def _word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _lexical_diversity(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def analyze_comment_text(
    text: str | None,
    *,
    template_count: int,
    config: CommentBoilerplateConfig,
) -> dict[str, Any]:
    raw_text = str(text or "").strip()
    markup_tags = _markup_tags(raw_text)
    plain_text = strip_pgn_markup(raw_text)
    normalized_plain = normalized_template_text(raw_text)
    tokens = _word_tokens(plain_text)
    lexical_diversity = _lexical_diversity(tokens)
    markup_char_count = sum(len(match.group(0)) for match in MARKUP_RE.finditer(raw_text))
    markup_char_share = (markup_char_count / len(raw_text)) if raw_text else 0.0

    categories: list[str] = []
    if raw_text and markup_tags and not plain_text:
        categories.append("markup_only")
    if markup_tags and (
        not plain_text
        or markup_char_share >= config.markup_heavy_share_threshold
        or (len(tokens) <= 2 and any(tag in {"eval", "clk", "emt"} for tag in markup_tags))
    ):
        categories.append("pgn_markup_heavy")
    if plain_text and RESULT_COMMENT_RE.match(plain_text):
        categories.append("result_comment")
    if plain_text and SYMBOL_ONLY_RE.match(plain_text):
        categories.append("symbol_only")
    if plain_text and ENGINE_TEMPLATE_RE.match(plain_text):
        categories.append("engine_template")
    if normalized_plain and template_count >= config.repeated_template_min_count and (
        len(plain_text) <= config.short_template_char_limit or len(tokens) <= config.short_template_token_limit
    ):
        categories.append("short_repeated_template")
    if (
        normalized_plain
        and template_count >= config.repeated_template_min_count
        and tokens
        and len(tokens) <= config.low_lexical_token_limit
        and lexical_diversity <= config.low_lexical_diversity_max
    ):
        categories.append("low_lexical_diversity_repeated")

    return {
        "raw_text": raw_text,
        "plain_text": plain_text,
        "normalized_template": normalized_plain,
        "template_count": template_count,
        "markup_tags": markup_tags,
        "markup_char_share": markup_char_share,
        "token_count": len(tokens),
        "lexical_diversity": lexical_diversity,
        "categories": categories,
    }


def annotate_comment_rows(
    *,
    input_root: str | Path,
    config: CommentBoilerplateConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    boilerplate_config = config or CommentBoilerplateConfig()
    rows_by_split = _load_rows_by_split(input_root)
    template_counter: Counter[str] = Counter()
    for rows in rows_by_split.values():
        for row in rows:
            template = normalized_template_text(row.get("comment_text"))
            if template:
                template_counter[template] += 1

    for rows in rows_by_split.values():
        for row in rows:
            analysis = analyze_comment_text(
                row.get("comment_text"),
                template_count=template_counter.get(normalized_template_text(row.get("comment_text")), 0),
                config=boilerplate_config,
            )
            row["comment_boilerplate"] = analysis
    return rows_by_split


def generate_comment_boilerplate_audit(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_v1",
    config: CommentBoilerplateConfig | None = None,
) -> dict[str, Any]:
    boilerplate_config = config or CommentBoilerplateConfig()
    rows_by_split = annotate_comment_rows(input_root=input_root, config=boilerplate_config)
    all_rows = [row for rows in rows_by_split.values() for row in rows]

    categories = (
        "markup_only",
        "pgn_markup_heavy",
        "result_comment",
        "symbol_only",
        "engine_template",
        "short_repeated_template",
        "low_lexical_diversity_repeated",
    )
    counts_by_category = {category: 0 for category in categories}
    share_by_split = {
        split_name: {category: 0 for category in categories}
        for split_name in ("train", "val", "test")
    }
    share_by_source: dict[str, dict[str, int]] = defaultdict(lambda: {category: 0 for category in categories})
    rows_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for split_name, rows in rows_by_split.items():
        for row in rows:
            analysis = row["comment_boilerplate"]
            categories_hit = list(analysis["categories"])
            for category in categories_hit:
                counts_by_category[category] += 1
                share_by_split[split_name][category] += 1
                source_name = str(row.get("comment_source") or "unknown")
                share_by_source[source_name][category] += 1
                rows_by_category[category].append(row)

    total_rows = len(all_rows)
    category_examples: dict[str, list[dict[str, Any]]] = {}
    top_repeated_strings: dict[str, list[dict[str, Any]]] = {}
    for category in categories:
        example_rows = rows_by_category[category][:10]
        category_examples[category] = [
            {
                "sidecar_id": row.get("sidecar_id"),
                "split": row.get("split"),
                "comment_source": row.get("comment_source"),
                "comment_text": row.get("comment_text"),
            }
            for row in example_rows
        ]
        counter: Counter[str] = Counter(
            str(row["comment_boilerplate"]["raw_text"])
            for row in rows_by_category[category]
            if str(row["comment_boilerplate"]["raw_text"])
        )
        top_repeated_strings[category] = [
            {"comment_text": text, "count": count}
            for text, count in counter.most_common(10)
        ]

    report = {
        "input_root": str(input_root),
        "config": asdict(boilerplate_config),
        "total_rows": total_rows,
        "rows_by_split": {split_name: len(rows) for split_name, rows in rows_by_split.items()},
        "counts_by_boilerplate_category": counts_by_category,
        "share_by_split": {
            split_name: {
                category: (count / len(rows_by_split[split_name])) if rows_by_split[split_name] else 0.0
                for category, count in category_counts.items()
            }
            for split_name, category_counts in share_by_split.items()
        },
        "share_by_comment_source": {
            source_name: {
                category: (count / total_source_rows) if total_source_rows else 0.0
                for category, count in category_counts.items()
            }
            for source_name, category_counts in sorted(share_by_source.items())
            for total_source_rows in [sum(1 for row in all_rows if str(row.get("comment_source") or "unknown") == source_name)]
        },
        "top_repeated_boilerplate_strings": top_repeated_strings,
        "examples_by_category": category_examples,
    }
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Comment Boilerplate Audit", ""]
    lines.append(f"- total_rows: {report['total_rows']}")
    lines.append(
        "- rows_by_split: "
        + ", ".join(f"{split_name}={count}" for split_name, count in report["rows_by_split"].items())
    )
    lines.append("")
    lines.append("## Counts By Category")
    for category, count in report["counts_by_boilerplate_category"].items():
        lines.append(f"- `{category}`: {count}")
    lines.append("")
    lines.append("## Share By Split")
    for split_name, category_counts in report["share_by_split"].items():
        formatted = ", ".join(f"{category}={share:.4f}" for category, share in category_counts.items() if share > 0.0)
        lines.append(f"- `{split_name}`: {formatted or 'none'}")
    lines.append("")
    lines.append("## Share By Comment Source")
    for source_name, category_counts in report["share_by_comment_source"].items():
        formatted = ", ".join(f"{category}={share:.4f}" for category, share in category_counts.items() if share > 0.0)
        lines.append(f"- `{source_name}`: {formatted or 'none'}")
    lines.append("")
    lines.append("## Top Boilerplate Strings")
    for category, rows in report["top_repeated_boilerplate_strings"].items():
        if not rows:
            continue
        lines.append(f"- `{category}`:")
        for row in rows[:5]:
            lines.append(f"  - `{row['comment_text']}`: {row['count']}")
    return "\n".join(lines) + "\n"


def write_comment_boilerplate_audit(
    *,
    input_root: str | Path = "data/pilot/annotated_sidecar_v1",
    output_dir: str | Path | None = None,
    config: CommentBoilerplateConfig | None = None,
) -> dict[str, str]:
    input_path = Path(input_root)
    report_root = Path(output_dir) if output_dir is not None else input_path / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    report = generate_comment_boilerplate_audit(input_root=input_path, config=config)
    json_path = report_root / "comment_boilerplate_audit.json"
    md_path = report_root / "comment_boilerplate_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return {"report_json": str(json_path), "report_md": str(md_path)}
