"""1주차 실험 결과를 집계해 표 형태로 내보낸다."""

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
)

TEST_METRIC_COLUMNS = (
    ("occupied_square_accuracy", "occ_acc"),
    ("piece_macro_f1", "piece_f1"),
    ("legality_average_precision", "legal_ap"),
    ("legality_f1", "legal_f1"),
)

SUBSET_NAMES = ("promotion", "castling", "en_passant", "check_evasion")
SUBSET_METRICS = ("top_1", "top_3", "top_5", "target_move_nll")


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _find_eval_summaries(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.rglob("eval_summary.json")
        if path.parent.name == "eval"
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _experiment_name(summary_path: Path, input_root: Path) -> str:
    relative = summary_path.relative_to(input_root)
    if not relative.parts:
        return "unknown"
    return relative.parts[0]


def _build_run_row(summary: dict[str, Any], summary_path: Path, input_root: Path) -> dict[str, object]:
    test_metrics = summary.get("splits", {}).get("test", {})
    row: dict[str, object] = {
        "experiment": _experiment_name(summary_path, input_root),
        "model": summary.get("model_type", "unknown"),
        "seed": summary.get("seed"),
    }
    for source_key, target_key in RUN_COLUMNS:
        row[target_key] = summary.get(source_key, 0.0)
    for source_key, target_key in TEST_METRIC_COLUMNS:
        row[target_key] = test_metrics.get(source_key, 0.0)
    return row


def _build_subset_rows(
    summary: dict[str, Any],
    summary_path: Path,
    input_root: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    experiment_name = _experiment_name(summary_path, input_root)
    for split_name, split_metrics in summary.get("splits", {}).items():
        subsets = split_metrics.get("subsets", {})
        for subset_name in SUBSET_NAMES:
            subset_metrics = subsets.get(subset_name, {})
            row: dict[str, object] = {
                "experiment": experiment_name,
                "model": summary.get("model_type", "unknown"),
                "seed": summary.get("seed"),
                "split": split_name,
                "subset": subset_name,
                "count": subset_metrics.get("count", 0),
            }
            for metric_name in SUBSET_METRICS:
                row[metric_name] = subset_metrics.get(metric_name, 0.0)
            rows.append(row)
    return rows


def _aggregate_run_rows(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in run_rows:
        key = (str(row["experiment"]), str(row["model"]))
        grouped.setdefault(key, []).append(row)
    aggregate_rows: list[dict[str, object]] = []
    for (experiment_name, model_name), rows in grouped.items():
        aggregate_row: dict[str, object] = {
            "experiment": experiment_name,
            "model": model_name,
            "seed": "mean±std",
        }
        metric_keys = [target_key for _, target_key in RUN_COLUMNS] + [target_key for _, target_key in TEST_METRIC_COLUMNS]
        for metric_key in metric_keys:
            mean_value, std_value = _mean_std([float(row[metric_key]) for row in rows])
            aggregate_row[metric_key] = f"{mean_value:.6f} ± {std_value:.6f}"
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _aggregate_subset_rows(subset_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in subset_rows:
        key = (
            str(row["experiment"]),
            str(row["model"]),
            str(row["split"]),
            str(row["subset"]),
        )
        grouped.setdefault(key, []).append(row)
    aggregate_rows: list[dict[str, object]] = []
    for (experiment_name, model_name, split_name, subset_name), rows in grouped.items():
        aggregate_row: dict[str, object] = {
            "experiment": experiment_name,
            "model": model_name,
            "seed": "mean±std",
            "split": split_name,
            "subset": subset_name,
            "count": sum(int(row["count"]) for row in rows),
        }
        for metric_name in SUBSET_METRICS:
            mean_value, std_value = _mean_std([float(row[metric_name]) for row in rows])
            aggregate_row[metric_name] = f"{mean_value:.6f} ± {std_value:.6f}"
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_week1(input_root: str | Path, output_dir: str | Path | None = None) -> dict[str, str]:
    """week1 eval_summary 산출물을 읽어 aggregate 표를 기록한다."""
    root = Path(input_root)
    destination = Path(output_dir) if output_dir is not None else root
    summary_paths = _find_eval_summaries(root)
    run_rows = [_build_run_row(_load_json(path), path, root) for path in summary_paths]
    subset_rows = []
    for path in summary_paths:
        subset_rows.extend(_build_subset_rows(_load_json(path), path, root))

    aggregate_payload = {
        "runs": run_rows,
        "aggregates": _aggregate_run_rows(run_rows),
    }
    subset_payload = {
        "runs": subset_rows,
        "aggregates": _aggregate_subset_rows(subset_rows),
    }

    aggregate_json = destination / "week1_aggregate.json"
    aggregate_csv = destination / "week1_aggregate.csv"
    subset_json = destination / "week1_subset.json"
    subset_csv = destination / "week1_subset.csv"

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
    parser.add_argument("--input-root", default="outputs/week1")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_week1(args.input_root, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
