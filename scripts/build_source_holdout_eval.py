"""Build source/source-family holdout evaluation regimes for annotated comments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.source_holdout_eval import (
    HoldoutThresholds,
    SourceHoldoutEvalConfig,
    build_source_holdout_eval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/pilot/annotated_sidecar_eval_v5")
    parser.add_argument("--output-root", default="data/pilot/annotated_sidecar_holdout_v1")
    parser.add_argument("--coarse-min-total", type=int, default=1000)
    parser.add_argument("--coarse-min-train", type=int, default=1000)
    parser.add_argument("--coarse-min-val", type=int, default=100)
    parser.add_argument("--coarse-min-test", type=int, default=100)
    parser.add_argument("--family-min-total", type=int, default=1500)
    parser.add_argument("--family-min-train", type=int, default=1000)
    parser.add_argument("--family-min-val", type=int, default=100)
    parser.add_argument("--family-min-test", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_source_holdout_eval(
        input_root=args.input_root,
        output_root=args.output_root,
        config=SourceHoldoutEvalConfig(
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
        ),
    )
    print(f"Wrote holdout manifest to {result['manifest_path']}")


if __name__ == "__main__":
    main()
