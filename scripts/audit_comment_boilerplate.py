"""Audit boilerplate and engine-style comment patterns in annotated PGN sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.comment_boilerplate_audit import write_comment_boilerplate_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/pilot/annotated_sidecar_v1")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_comment_boilerplate_audit(
        input_root=args.input_root,
        output_dir=args.output_dir,
    )
    print(f"Wrote comment boilerplate audit to {result['report_json']}")


if __name__ == "__main__":
    main()
