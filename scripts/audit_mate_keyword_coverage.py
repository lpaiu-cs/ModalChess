"""Audit conservative MATE keyword densification candidates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.mate_keyword_audit import audit_mate_keyword_coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", default="data/pilot/real_v1/language_mate.jsonl")
    parser.add_argument("--output-dir", default="data/pilot/language_probe_v3/reports")
    parser.add_argument("--min-support", type=int, default=50)
    parser.add_argument("--max-examples-per-label", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_mate_keyword_coverage(
        input_path=args.input_path,
        output_dir=args.output_dir,
        min_support=args.min_support,
        max_examples_per_label=args.max_examples_per_label,
    )
    print(f"Wrote MATE keyword audit to {result['json_path']}")


if __name__ == "__main__":
    main()
