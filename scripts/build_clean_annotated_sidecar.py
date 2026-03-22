"""Build a cleaner filtered annotated PGN comment sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.clean_annotated_sidecar import (
    CleanAnnotatedSidecarConfig,
    build_clean_annotated_sidecar,
)
from modalchess.data.comment_boilerplate_audit import CommentBoilerplateConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/pilot/annotated_sidecar_v1")
    parser.add_argument("--output-root", default="data/pilot/annotated_sidecar_v2_clean")
    parser.add_argument("--primary-variant", default="keep_comment_source_balance")
    parser.add_argument("--template-cap-per-cluster", type=int, default=3)
    parser.add_argument("--template-cap-per-source-cluster", type=int, default=3)
    parser.add_argument("--repeated-template-min-count", type=int, default=20)
    parser.add_argument("--short-template-char-limit", type=int, default=24)
    parser.add_argument("--short-template-token-limit", type=int, default=4)
    parser.add_argument("--low-lexical-diversity-max", type=float, default=0.50)
    parser.add_argument("--low-lexical-token-limit", type=int, default=6)
    parser.add_argument("--markup-heavy-share-threshold", type=float, default=0.35)
    parser.add_argument("--salt", default="modalchess_week13_clean_sidecar")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_clean_annotated_sidecar(
        input_root=args.input_root,
        output_root=args.output_root,
        config=CleanAnnotatedSidecarConfig(
            primary_variant=args.primary_variant,
            boilerplate_config=CommentBoilerplateConfig(
                repeated_template_min_count=args.repeated_template_min_count,
                short_template_char_limit=args.short_template_char_limit,
                short_template_token_limit=args.short_template_token_limit,
                low_lexical_diversity_max=args.low_lexical_diversity_max,
                low_lexical_token_limit=args.low_lexical_token_limit,
                markup_heavy_share_threshold=args.markup_heavy_share_threshold,
            ),
            template_cap_per_cluster=args.template_cap_per_cluster,
            template_cap_per_source_cluster=args.template_cap_per_source_cluster,
            salt=args.salt,
        ),
    )
    print(f"Wrote clean sidecar manifest to {result['manifest_path']}")


if __name__ == "__main__":
    main()
