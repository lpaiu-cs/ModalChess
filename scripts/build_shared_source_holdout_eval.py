"""Build shared comparable source-holdout regimes across week-17 variants."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.shared_source_holdout_eval import (
    SharedSourceHoldoutEvalConfig,
    build_shared_source_holdout_eval,
)
from modalchess.data.source_holdout_eval import HoldoutThresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant-root",
        action="append",
        default=[],
        help="name=path form; repeatable",
    )
    parser.add_argument("--output-root", default="data/pilot/annotated_sidecar_holdout_v2")
    parser.add_argument("--coarse-min-total", type=int, default=300)
    parser.add_argument("--coarse-min-train", type=int, default=150)
    parser.add_argument("--coarse-min-val", type=int, default=20)
    parser.add_argument("--coarse-min-test", type=int, default=20)
    parser.add_argument("--family-min-total", type=int, default=300)
    parser.add_argument("--family-min-train", type=int, default=150)
    parser.add_argument("--family-min-val", type=int, default=20)
    parser.add_argument("--family-min-test", type=int, default=20)
    parser.add_argument("--min-shared-test-rows", type=int, default=100)
    return parser.parse_args()


def _parse_variant_roots(values: list[str]) -> dict[str, str]:
    variant_roots: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--variant-root must use name=path form: {value}")
        name, path = value.split("=", 1)
        variant_roots[name] = path
    return variant_roots


def main() -> None:
    args = parse_args()
    variant_roots = _parse_variant_roots(args.variant_root)
    result = build_shared_source_holdout_eval(
        variant_roots=variant_roots,
        output_root=args.output_root,
        config=SharedSourceHoldoutEvalConfig(
            coarse_thresholds=HoldoutThresholds(
                min_total_rows=args.coarse_min_total,
                min_train_rows=args.coarse_min_train,
                min_val_rows=args.coarse_min_val,
                min_test_rows=args.coarse_min_test,
            ),
            family_thresholds=HoldoutThresholds(
                min_total_rows=args.family_min_total,
                min_train_rows=args.family_min_train,
                min_val_rows=args.family_min_val,
                min_test_rows=args.family_min_test,
            ),
            min_shared_test_rows=args.min_shared_test_rows,
        ),
    )
    print(f"Wrote shared holdout manifest to {result['manifest_path']}")


if __name__ == "__main__":
    main()
