"""Inventory locally available annotated comment sources for week-15."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.annotated_source_fetch import fetch_more_annotated_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/pilot/annotated_sidecar_v2_sources")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = fetch_more_annotated_sources(output_root=args.output_root)
    print(f"Wrote source fetch lock to {result['lock_path']}")


if __name__ == "__main__":
    main()
