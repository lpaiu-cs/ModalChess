"""Build rationale-ready sidecar JSONL files for week-4."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.rationale_sidecar import RationaleBuildConfig, build_rationale_sidecars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-rationale-chars", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_rationale_sidecars(
        input_root=args.input_root,
        output_root=args.output_root,
        config=RationaleBuildConfig(max_rationale_chars=args.max_rationale_chars),
    )
    print(f"Wrote rationale manifest to {result['manifest_path']}")


if __name__ == "__main__":
    main()
