"""Write QA reports for week-4 language sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.language_sidecar_report import write_language_sidecar_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_language_sidecar_report(
        input_root=args.input_root,
        output_dir=args.output_dir,
    )
    print(f"Wrote language sidecar report to {result['json']} and {result['md']}")


if __name__ == "__main__":
    main()
