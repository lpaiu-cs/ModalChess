"""구조화된 리포트 작성 유틸리티."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def flatten_metrics(
    metrics: dict[str, Any],
    prefix: str = "",
) -> dict[str, int | float | str | bool]:
    """중첩된 metric 딕셔너리를 CSV 기록용 평탄 구조로 변환한다."""
    flattened: dict[str, int | float | str | bool] = {}
    for key, value in metrics.items():
        flat_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, prefix=flat_key))
            continue
        if isinstance(value, (int, float, str, bool)):
            flattened[flat_key] = value
    return flattened


def write_report(metrics: dict[str, Any], output_dir: str | Path, name: str = "eval_report") -> dict[str, str]:
    """JSON 및 CSV 형식의 평가 리포트를 기록한다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}.json"
    csv_path = output_dir / f"{name}.csv"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    flattened = flatten_metrics(metrics)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in flattened.items():
            writer.writerow([key, value])
    return {"json": str(json_path), "csv": str(csv_path)}


def write_failure_dump(
    rows: list[dict[str, object]],
    output_dir: str | Path,
    name: str = "eval_failures",
) -> str:
    """sample-level failure 사례를 JSONL로 기록한다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(jsonl_path)
