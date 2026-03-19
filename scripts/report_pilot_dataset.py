"""real pilot supervised dataset의 QA report를 생성한다."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.pilot_report import write_pilot_data_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="real pilot dataset root")
    parser.add_argument("--output-dir", required=True, help="report output dir")
    parser.add_argument("--min-val-rows", type=int, default=1000)
    parser.add_argument("--min-test-rows", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_pilot_data_report(
        input_root=args.input_root,
        output_dir=args.output_dir,
        min_val_rows=args.min_val_rows,
        min_test_rows=args.min_test_rows,
    )
    print(f"Wrote pilot data report to {paths['json']} and {paths['md']}")


if __name__ == "__main__":
    main()
