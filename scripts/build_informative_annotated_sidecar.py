"""Build a higher-information annotated comment sidecar for evaluation-only retrieval.""" 

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.comment_informativeness import CommentInformativenessConfig
from modalchess.data.informative_annotated_sidecar import (
    InformativeAnnotatedSidecarConfig,
    build_informative_annotated_sidecar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/pilot/annotated_sidecar_v2_clean")
    parser.add_argument("--output-root", default="data/pilot/annotated_sidecar_v3_informative")
    parser.add_argument("--primary-variant", default="medium_high_only")
    parser.add_argument("--low-threshold", type=float, default=0.33)
    parser.add_argument("--medium-threshold", type=float, default=0.45)
    parser.add_argument("--high-threshold", type=float, default=0.62)
    parser.add_argument("--medium-source-quantile", type=float, default=0.60)
    parser.add_argument("--special-rule-floor", type=float, default=0.30)
    parser.add_argument("--salt", default="modalchess_week14_informative_sidecar")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_informative_annotated_sidecar(
        input_root=args.input_root,
        output_root=args.output_root,
        config=InformativeAnnotatedSidecarConfig(
            primary_variant=args.primary_variant,
            informativeness_config=CommentInformativenessConfig(
                low_threshold=args.low_threshold,
                medium_threshold=args.medium_threshold,
                high_threshold=args.high_threshold,
                medium_source_quantile=args.medium_source_quantile,
                special_rule_floor=args.special_rule_floor,
            ),
            salt=args.salt,
        ),
    )
    print(f"Wrote informative sidecar manifest to {result['manifest_path']}")


if __name__ == "__main__":
    main()
