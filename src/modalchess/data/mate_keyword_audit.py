"""Audit conservative MATE keyword-map densification candidates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from modalchess.data.preprocessing_common import iter_records_from_path
from modalchess.data.probe_targets import MATE_KEYWORD_MAP


@dataclass(frozen=True, slots=True)
class KeywordAuditCandidate:
    label: str
    patterns: tuple[str, ...]
    ambiguity_risk: str
    likely_false_positive_patterns: tuple[str, ...]
    rationale: str


CANDIDATE_KEYWORD_EXPANSIONS: tuple[KeywordAuditCandidate, ...] = (
    KeywordAuditCandidate(
        label="trade_up",
        patterns=(
            "trade the lower value piece for a higher value piece",
            "trade your lesser piece for a more valuable piece",
            "more valuable piece",
        ),
        ambiguity_risk="low",
        likely_false_positive_patterns=(),
        rationale="Captures a specific tactical exchange motif that current `capture` is too broad to isolate.",
    ),
    KeywordAuditCandidate(
        label="sacrifice_for_attack",
        patterns=(
            "sacrifice a piece",
            "surrender a piece",
        ),
        ambiguity_risk="medium",
        likely_false_positive_patterns=("generic strategic sacrifice",),
        rationale="Frequent explicit phrasing for line-opening or king-attack ideas, but sacrifice language can be strategically broad.",
    ),
    KeywordAuditCandidate(
        label="open_line_attack",
        patterns=(
            "open file or diagonal",
            "unlock a file or diagonal",
            "create an open file or diagonal",
        ),
        ambiguity_risk="low",
        likely_false_positive_patterns=("non-attacking open-file plans",),
        rationale="Combines open-line creation with attack-oriented language more specifically than the current `open_file` label.",
    ),
    KeywordAuditCandidate(
        label="piece_activity",
        patterns=(
            "more actively",
            "greater board command",
            "more influence over the board",
            "control over the board",
        ),
        ambiguity_risk="high",
        likely_false_positive_patterns=("generic strategic improvement", "non-tactical maneuvering"),
        rationale="Very common in MATE texts, but highly strategic and semantically broad.",
    ),
    KeywordAuditCandidate(
        label="king_attack_zone",
        patterns=(
            "opposing king",
            "enemy king",
            "near the opposing king",
            "proximity to the opposing king",
        ),
        ambiguity_risk="medium",
        likely_false_positive_patterns=("non-forcing king-side improvement",),
        rationale="Signals king-focused pressure, but overlaps partially with `king_safety`.",
    ),
)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _snippet(text: str, pattern: str, window: int = 120) -> str:
    lowered = text.lower()
    index = lowered.find(pattern.lower())
    if index < 0:
        return _clean_text(text[:window])
    start = max(0, index - window // 2)
    end = min(len(text), index + len(pattern) + window // 2)
    return _clean_text(text[start:end])


def audit_mate_keyword_coverage(
    *,
    input_path: str | Path = "data/pilot/real_v1/language_mate.jsonl",
    output_dir: str | Path = "data/pilot/language_probe_v3/reports",
    min_support: int = 50,
    max_examples_per_label: int = 5,
) -> dict[str, Any]:
    """Audit candidate conservative MATE keyword expansions."""
    path = Path(input_path)
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    existing_support = Counter()
    candidate_support = Counter()
    candidate_examples: dict[str, list[str]] = {candidate.label: [] for candidate in CANDIDATE_KEYWORD_EXPANSIONS}
    total_rows = 0

    for row in iter_records_from_path(path):
        total_rows += 1
        text = " ".join(str(row.get(key) or "") for key in ("strategy_text", "tactic_text"))
        lowered = text.lower()
        for label_name, keywords in MATE_KEYWORD_MAP.items():
            if any(keyword in lowered for keyword in keywords):
                existing_support[label_name] += 1
        for candidate in CANDIDATE_KEYWORD_EXPANSIONS:
            matched_pattern = next((pattern for pattern in candidate.patterns if pattern in lowered), None)
            if matched_pattern is None:
                continue
            candidate_support[candidate.label] += 1
            if len(candidate_examples[candidate.label]) < max_examples_per_label:
                candidate_examples[candidate.label].append(_snippet(text, matched_pattern))

    candidate_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATE_KEYWORD_EXPANSIONS:
        support_count = candidate_support[candidate.label]
        candidate_rows.append(
            {
                "label": candidate.label,
                "support_count": support_count,
                "patterns": list(candidate.patterns),
                "ambiguity_risk": candidate.ambiguity_risk,
                "likely_false_positive_patterns": list(candidate.likely_false_positive_patterns),
                "examples": candidate_examples[candidate.label],
                "rationale": candidate.rationale,
                "recommended_for_future_v2": support_count >= min_support and candidate.ambiguity_risk != "high",
            }
        )

    report = {
        "input_path": str(path),
        "total_rows": total_rows,
        "existing_label_support": dict(sorted(existing_support.items())),
        "candidate_labels": candidate_rows,
        "conclusions": [
            "trade_up and open_line_attack have strong support and relatively constrained phrasing, so they are the safest densification candidates.",
            "sacrifice_for_attack has meaningful support but requires care because sacrifice language can be strategic rather than tactical.",
            "piece_activity is frequent but too broad to auto-promote without heavier semantic filtering.",
            "This audit is advisory only; week-7 does not silently replace the week-6 target set.",
        ],
    }

    json_path = report_dir / "mate_keyword_audit.json"
    md_path = report_dir / "mate_keyword_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "report": report,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# MATE Keyword Audit", ""]
    lines.append(f"- total_rows: {report['total_rows']}")
    lines.append("")
    lines.append("## Candidate Labels")
    for candidate in report["candidate_labels"]:
        lines.append(
            f"- `{candidate['label']}`: support={candidate['support_count']}, "
            f"ambiguity_risk={candidate['ambiguity_risk']}, "
            f"recommended_for_future_v2={candidate['recommended_for_future_v2']}"
        )
        for example in candidate["examples"]:
            lines.append(f"  example: {example}")
    lines.append("")
    lines.append("## Conclusions")
    for conclusion in report["conclusions"]:
        lines.append(f"- {conclusion}")
    return "\n".join(lines) + "\n"
