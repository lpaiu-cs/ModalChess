"""Build conservative targets for week-5 standalone probe corpora."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.probe_targets import ProbeTargetConfig, build_probe_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rare-label-threshold", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_probe_targets(
        input_root=args.input_root,
        output_root=args.output_root,
        config=ProbeTargetConfig(rare_label_threshold=args.rare_label_threshold),
    )
    print(f"Wrote probe target manifest to {result['manifest_path']}")


if __name__ == "__main__":
    main()
