"""Build dedup-aware evaluation corpora from annotated comment sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.dedup_comment_eval import DedupCommentEvalConfig, build_dedup_comment_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/pilot/annotated_sidecar_v1")
    parser.add_argument("--output-root", default="data/pilot/annotated_sidecar_eval_v2")
    parser.add_argument("--primary-variant", default="normalized_comment_dedup")
    parser.add_argument("--normalized-mode", default="nag_stripped")
    parser.add_argument("--capped-duplicates-per-cluster", type=int, default=3)
    parser.add_argument("--salt", default="modalchess_week12_dedup_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dedup_comment_eval(
        input_root=args.input_root,
        output_root=args.output_root,
        config=DedupCommentEvalConfig(
            primary_variant=args.primary_variant,
            normalized_mode=args.normalized_mode,
            capped_duplicates_per_cluster=args.capped_duplicates_per_cluster,
            salt=args.salt,
        ),
    )
    print(f"Wrote dedup eval manifest to {result['manifest_path']}")


if __name__ == "__main__":
    main()
