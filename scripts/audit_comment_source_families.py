"""Audit source-family composition of annotated comment corpora."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.comment_informativeness import CommentInformativenessConfig
from modalchess.data.comment_source_audit import CommentSourceAuditConfig, write_comment_source_family_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/pilot/annotated_sidecar_v3_informative")
    parser.add_argument("--output-dir")
    parser.add_argument("--low-threshold", type=float, default=0.33)
    parser.add_argument("--medium-threshold", type=float, default=0.45)
    parser.add_argument("--high-threshold", type=float, default=0.62)
    parser.add_argument("--medium-source-quantile", type=float, default=0.60)
    parser.add_argument("--special-rule-floor", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CommentSourceAuditConfig(
        informativeness_config=CommentInformativenessConfig(
            low_threshold=args.low_threshold,
            medium_threshold=args.medium_threshold,
            high_threshold=args.high_threshold,
            medium_source_quantile=args.medium_source_quantile,
            special_rule_floor=args.special_rule_floor,
        )
    )
    result = write_comment_source_family_audit(
        input_root=args.input_root,
        output_dir=args.output_dir,
        config=config,
    )
    print(f"Wrote source family audit to {result['json_path']}")


if __name__ == "__main__":
    main()
