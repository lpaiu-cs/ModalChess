"""Lock frozen week-4 G1/G3 reference artifacts without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.eval.week4_reference import lock_week4_reference_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week3-root", default="outputs/week3")
    parser.add_argument("--output-dir", default="outputs/week4/reference_artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = lock_week4_reference_artifacts(
        week3_root=args.week3_root,
        output_dir=args.output_dir,
    )
    print(f"Wrote reference summary to {result['summary_path']}")


if __name__ == "__main__":
    main()
