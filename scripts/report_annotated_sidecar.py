"""Report QA and week-8 comparison for the annotated PGN sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.annotated_pgn_sidecar import write_annotated_sidecar_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/pilot/annotated_sidecar_v1")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--compare-aux-root", default="data/pilot/language_probe_v4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_annotated_sidecar_report(
        input_root=args.input_root,
        output_dir=args.output_dir,
        compare_aux_root=args.compare_aux_root,
    )
    print(f"Wrote annotated sidecar report to {result['report_json']}")


if __name__ == "__main__":
    main()
