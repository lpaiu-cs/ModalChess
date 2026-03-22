"""Audit comment informativeness on the cleaned annotated sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.comment_informativeness import (
    CommentInformativenessConfig,
    write_comment_informativeness_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/pilot/annotated_sidecar_v2_clean")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--low-threshold", type=float, default=0.33)
    parser.add_argument("--medium-threshold", type=float, default=0.45)
    parser.add_argument("--high-threshold", type=float, default=0.62)
    parser.add_argument("--medium-source-quantile", type=float, default=0.60)
    parser.add_argument("--special-rule-floor", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_comment_informativeness_audit(
        input_root=args.input_root,
        output_dir=args.output_dir,
        config=CommentInformativenessConfig(
            low_threshold=args.low_threshold,
            medium_threshold=args.medium_threshold,
            high_threshold=args.high_threshold,
            medium_source_quantile=args.medium_source_quantile,
            special_rule_floor=args.special_rule_floor,
        ),
    )
    print(f"Wrote comment informativeness audit to {result['report_json']}")


if __name__ == "__main__":
    main()
