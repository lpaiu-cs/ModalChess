"""Week-3 grounding ablation 결과를 집계한다."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any


RUN_COLUMNS = (
    ("val_target_move_nll", "val_nll"),
    ("val_top_1", "val_top1"),
    ("val_top_3", "val_top3"),
    ("val_top_5", "val_top5"),
    ("test_target_move_nll", "test_nll"),
    ("test_top_1", "test_top1"),
    ("test_top_3", "test_top3"),
    ("test_top_5", "test_top5"),
    ("puzzle_target_move_nll", "puzzle_nll"),
    ("puzzle_top_1", "puzzle_top1"),
    ("puzzle_top_3", "puzzle_top3"),
    ("puzzle_top_5", "puzzle_top5"),
)

TEST_METRIC_COLUMNS = (
    ("test_occupied_square_accuracy", "occ_acc"),
    ("test_piece_macro_f1", "piece_f1"),
    ("test_legality_average_precision", "legal_ap"),
    ("test_legality_f1", "legal_f1"),
)

PARAMETER_COLUMNS = (
    ("model_parameter_count", "model_parameter_count"),
    ("trainable_parameter_count", "trainable_parameter_count"),
)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _collect_main_summaries(input_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(input_root.glob("exp3_ground_*/seed*/eval_main/eval_summary.json")):
        experiment = path.parents[2].name
        seed = path.parents[1].name.removeprefix("seed")
        summaries[(experiment, seed)] = _load_json(path)
    return summaries


def _collect_puzzle_summaries(puzzle_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(puzzle_root.glob("exp3_ground_*/seed*/eval_summary.json")):
        experiment = path.parents[1].name
        seed = path.parent.name.removeprefix("seed")
        summaries[(experiment, seed)] = _load_json(path)
    return summaries


def _build_run_rows(
    main_summaries: dict[tuple[str, str], dict[str, Any]],
    puzzle_summaries: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, summary in sorted(main_summaries.items()):
        experiment, seed = key
        merged_summary = dict(summary)
        puzzle_summary = puzzle_summaries.get(key)
        if puzzle_summary is not None:
            puzzle_metrics = puzzle_summary.get("splits", {}).get("puzzle", {})
            for metric_name in ("target_move_nll", "top_1", "top_3", "top_5"):
                merged_summary[f"puzzle_{metric_name}"] = puzzle_metrics.get(metric_name, 0.0)

        row: dict[str, object] = {
            "experiment": experiment,
            "model": merged_summary.get("model_type", "unknown"),
            "seed": seed,
        }
        for source_key, target_key in RUN_COLUMNS:
            row[target_key] = merged_summary.get(source_key, 0.0)
        for source_key, target_key in TEST_METRIC_COLUMNS:
            row[target_key] = merged_summary.get(source_key, 0.0)
        for source_key, target_key in PARAMETER_COLUMNS:
            row[target_key] = int(merged_summary.get(source_key, 0))
        rows.append(row)
    return rows


def _build_subset_rows(
    main_summaries: dict[tuple[str, str], dict[str, Any]],
    puzzle_summaries: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    merged_by_run = dict(main_summaries)
    for key, puzzle_summary in puzzle_summaries.items():
        if key not in merged_by_run:
            merged_by_run[key] = {"splits": {}}
        merged_by_run[key].setdefault("splits", {}).update(puzzle_summary.get("splits", {}))

    for (experiment, seed), summary in sorted(merged_by_run.items()):
        for split_name, split_metrics in summary.get("splits", {}).items():
            subsets = split_metrics.get("subsets", {})
            for subset_name, subset_metrics in subsets.items():
                row: dict[str, object] = {
                    "experiment": experiment,
                    "model": summary.get("model_type", "unknown"),
                    "seed": seed,
                    "split": split_name,
                    "subset": subset_name,
                    "count": subset_metrics.get("count", 0),
                    "top_1": subset_metrics.get("top_1", 0.0),
                    "top_3": subset_metrics.get("top_3", 0.0),
                    "top_5": subset_metrics.get("top_5", 0.0),
                    "target_move_nll": subset_metrics.get("target_move_nll", 0.0),
                }
                rows.append(row)
    return rows


def _aggregate_rows(
    rows: list[dict[str, object]],
    group_keys: list[str],
    metric_keys: list[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(str(row[group_key]) for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for key, grouped_rows in grouped.items():
        aggregate_row: dict[str, object] = {
            group_key: key[index]
            for index, group_key in enumerate(group_keys)
        }
        aggregate_row["seed"] = "mean±std"
        if "count" in grouped_rows[0]:
            aggregate_row["count"] = sum(int(row["count"]) for row in grouped_rows)
        for metric_key in metric_keys:
            mean_value, std_value = _mean_std([float(row[metric_key]) for row in grouped_rows])
            aggregate_row[metric_key] = f"{mean_value:.6f} ± {std_value:.6f}"
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def aggregate_week3(
    input_root: str | Path = "outputs/week3",
    puzzle_root: str | Path = "outputs/week3/puzzle_aux_eval",
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    root = Path(input_root)
    puzzle = Path(puzzle_root)
    destination = Path(output_dir) if output_dir is not None else root
    main_summaries = _collect_main_summaries(root)
    puzzle_summaries = _collect_puzzle_summaries(puzzle)
    run_rows = _build_run_rows(main_summaries, puzzle_summaries)
    subset_rows = _build_subset_rows(main_summaries, puzzle_summaries)

    aggregate_payload = {
        "runs": run_rows,
        "aggregates": _aggregate_rows(
            run_rows,
            group_keys=["experiment", "model"],
            metric_keys=[
                target_key for _, target_key in RUN_COLUMNS
            ] + [target_key for _, target_key in TEST_METRIC_COLUMNS] + [target_key for _, target_key in PARAMETER_COLUMNS],
        ),
    }
    subset_payload = {
        "runs": subset_rows,
        "aggregates": _aggregate_rows(
            subset_rows,
            group_keys=["experiment", "model", "split", "subset"],
            metric_keys=["top_1", "top_3", "top_5", "target_move_nll"],
        ),
    }

    aggregate_json = destination / "exp3_aggregate.json"
    aggregate_csv = destination / "exp3_aggregate.csv"
    subset_json = destination / "exp3_subset.json"
    subset_csv = destination / "exp3_subset.csv"

    aggregate_json.write_text(json.dumps(aggregate_payload, indent=2), encoding="utf-8")
    subset_json.write_text(json.dumps(subset_payload, indent=2), encoding="utf-8")
    _write_csv(aggregate_csv, aggregate_payload["runs"] + aggregate_payload["aggregates"])
    _write_csv(subset_csv, subset_payload["runs"] + subset_payload["aggregates"])
    return {
        "aggregate_json": str(aggregate_json),
        "aggregate_csv": str(aggregate_csv),
        "subset_json": str(subset_json),
        "subset_csv": str(subset_csv),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="outputs/week3")
    parser.add_argument("--puzzle-root", default="outputs/week3/puzzle_aux_eval")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_week3(args.input_root, puzzle_root=args.puzzle_root, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
