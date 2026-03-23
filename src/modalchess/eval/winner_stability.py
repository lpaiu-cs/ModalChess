"""Winner-stability reporting across week-17 and week-18 retrieval regimes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WinnerRow:
    variant: str
    regime_name: str
    direction: str
    backbone: str
    pool: str
    probe_model: str
    metric_value: float
    seed: int | None = None


def _winner_by_metric(rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row[metric_name]),
            -float(row.get("test_rows_mean", row.get("test_rows", 0))),
            str(row["backbone"]),
        ),
    )


def _week17_winners(week17_payload: dict[str, Any]) -> list[WinnerRow]:
    winners: list[WinnerRow] = []
    for row in week17_payload.get("comparison", []):
        winners.append(
            WinnerRow(
                variant=str(row["variant"]),
                regime_name="week17_shared_pool",
                direction="board_to_text",
                backbone=str(row["best_board_backbone"]),
                pool=str(row["best_board_pool"]),
                probe_model=str(row["best_board_probe_model"]),
                metric_value=float(row["best_strict_board_to_text_mrr_mean"]),
            )
        )
        winners.append(
            WinnerRow(
                variant=str(row["variant"]),
                regime_name="week17_shared_pool",
                direction="text_to_board",
                backbone=str(row["best_text_backbone"]),
                pool=str(row["best_text_pool"]),
                probe_model=str(row["best_text_probe_model"]),
                metric_value=float(row["best_strict_text_to_board_mrr_mean"]),
            )
        )
    return winners


def _week18_regime_winners(week18_payload: dict[str, Any]) -> tuple[list[WinnerRow], list[WinnerRow]]:
    regime_winners: list[WinnerRow] = []
    seed_winners: list[WinnerRow] = []
    for variant, payload in (week18_payload.get("variants") or {}).items():
        aggregate_rows = payload.get("aggregate") or []
        results_rows = payload.get("results") or []
        by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in aggregate_rows:
            by_regime[str(row["regime_name"])].append(row)
        for regime_name, rows in by_regime.items():
            board_winner = _winner_by_metric(rows, "strict_board_to_text_mrr_mean")
            text_winner = _winner_by_metric(rows, "strict_text_to_board_mrr_mean")
            regime_winners.append(
                WinnerRow(
                    variant=str(variant),
                    regime_name=regime_name,
                    direction="board_to_text",
                    backbone=str(board_winner["backbone"]),
                    pool=str(board_winner["pool"]),
                    probe_model=str(board_winner["probe_model"]),
                    metric_value=float(board_winner["strict_board_to_text_mrr_mean"]),
                )
            )
            regime_winners.append(
                WinnerRow(
                    variant=str(variant),
                    regime_name=regime_name,
                    direction="text_to_board",
                    backbone=str(text_winner["backbone"]),
                    pool=str(text_winner["pool"]),
                    probe_model=str(text_winner["probe_model"]),
                    metric_value=float(text_winner["strict_text_to_board_mrr_mean"]),
                )
            )

        by_regime_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in results_rows:
            by_regime_seed[(str(row["regime_name"]), int(row["seed"]))].append(row)
        for (regime_name, seed), rows in by_regime_seed.items():
            board_winner = _winner_by_metric(rows, "strict_board_to_text_mrr")
            text_winner = _winner_by_metric(rows, "strict_text_to_board_mrr")
            seed_winners.append(
                WinnerRow(
                    variant=str(variant),
                    regime_name=regime_name,
                    direction="board_to_text",
                    backbone=str(board_winner["backbone"]),
                    pool=str(board_winner["pool"]),
                    probe_model=str(board_winner["probe_model"]),
                    metric_value=float(board_winner["strict_board_to_text_mrr"]),
                    seed=seed,
                )
            )
            seed_winners.append(
                WinnerRow(
                    variant=str(variant),
                    regime_name=regime_name,
                    direction="text_to_board",
                    backbone=str(text_winner["backbone"]),
                    pool=str(text_winner["pool"]),
                    probe_model=str(text_winner["probe_model"]),
                    metric_value=float(text_winner["strict_text_to_board_mrr"]),
                    seed=seed,
                )
            )
    return regime_winners, seed_winners


def build_winner_stability_report(
    *,
    week17_results_path: str | Path = "outputs/week17/comment_retrieval_v6/comment_retrieval_results.json",
    week18_holdout_path: str | Path = "outputs/week18/source_holdout_balanced/results.json",
    output_dir: str | Path = "outputs/week18/winner_stability",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    week17_payload = json.loads(Path(week17_results_path).read_text(encoding="utf-8"))
    week18_payload = json.loads(Path(week18_holdout_path).read_text(encoding="utf-8"))

    week17_winners = _week17_winners(week17_payload)
    regime_winners, seed_winners = _week18_regime_winners(week18_payload)

    regime_counter: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in regime_winners:
        regime_counter[row.variant][row.direction][row.backbone] += 1

    seed_counter: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in seed_winners:
        seed_counter[row.variant][row.direction][row.backbone] += 1

    payload = {
        "week17_shared_pool_winners": [
            {
                "variant": row.variant,
                "direction": row.direction,
                "backbone": row.backbone,
                "pool": row.pool,
                "probe_model": row.probe_model,
                "metric_value": row.metric_value,
            }
            for row in week17_winners
        ],
        "week18_regime_winner_counts": {
            variant: {
                direction: dict(counter)
                for direction, counter in directions.items()
            }
            for variant, directions in regime_counter.items()
        },
        "week18_seed_level_winner_counts": {
            variant: {
                direction: dict(counter)
                for direction, counter in directions.items()
            }
            for variant, directions in seed_counter.items()
        },
        "assessment": "regime_sensitive_backbone_rankings" if regime_counter else "insufficient_data",
    }
    json_path = output_path / "winner_stability.json"
    md_path = output_path / "winner_stability.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Winner Stability", ""]
    lines.append("## Week 17 Shared-Pool Winners")
    for row in week17_winners:
        lines.append(
            f"- `{row.variant}` / `{row.direction}`: `{row.backbone}` "
            f"({row.pool} / {row.probe_model}) = {row.metric_value:.6f}"
        )
    lines.append("")
    lines.append("## Week 18 Regime Winner Counts")
    for variant, directions in payload["week18_regime_winner_counts"].items():
        lines.append(f"### {variant}")
        for direction, counter in directions.items():
            lines.append(f"- `{direction}`: {counter}")
    lines.append("")
    lines.append("## Week 18 Seed-Level Winner Counts")
    for variant, directions in payload["week18_seed_level_winner_counts"].items():
        lines.append(f"### {variant}")
        for direction, counter in directions.items():
            lines.append(f"- `{direction}`: {counter}")
    lines.append("")
    lines.append(f"- assessment: `{payload['assessment']}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "payload": payload,
    }
