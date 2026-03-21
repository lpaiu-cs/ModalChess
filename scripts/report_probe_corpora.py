"""Write week-5 probe-corpora and rationale-readiness reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.probe_reports import write_probe_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compare-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_probe_reports(
        input_root=args.input_root,
        output_dir=args.output_dir,
        compare_root=args.compare_root,
    )
    print(
        "Wrote probe reports to "
        f"{result['probe_json']} / {result['probe_md']} and "
        f"{result['rationale_json']} / {result['rationale_md']}"
    )


if __name__ == "__main__":
    main()
