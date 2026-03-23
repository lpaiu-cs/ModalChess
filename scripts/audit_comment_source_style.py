"""Audit writing-style divergence across multisource annotated comment families."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.comment_source_style import (
    CommentSourceStyleAuditConfig,
    write_comment_source_style_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/pilot/annotated_sidecar_v4_multisource")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-group-rows", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_comment_source_style_audit(
        input_root=args.input_root,
        output_dir=args.output_dir,
        config=CommentSourceStyleAuditConfig(min_group_rows=args.min_group_rows),
    )
    print(f"Wrote source style audit to {result['report_json']}")


if __name__ == "__main__":
    main()
