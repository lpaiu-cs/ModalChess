"""Compare week-17 retrieval variants with paired query-level resampling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.eval.retrieval_comparison import (
    RetrievalComparisonConfig,
    compare_retrieval_variants,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="outputs/week17/comment_retrieval_v6")
    parser.add_argument("--embedding-root", default="outputs/week17/embedding_exports")
    parser.add_argument("--output-dir", default="outputs/week18/retrieval_comparison")
    parser.add_argument("--baseline-variant", default="current_mixed_baseline")
    parser.add_argument("--comparison-variant", action="append", default=[])
    parser.add_argument("--backbone-seed", type=int, action="append", dest="backbone_seeds", default=[])
    parser.add_argument("--mate-min-df", type=int, default=25)
    parser.add_argument("--max-vocab-size", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_retrieval_variants(
        results_root=args.results_root,
        embedding_root=args.embedding_root,
        output_dir=args.output_dir,
        baseline_variant=args.baseline_variant,
        comparison_variants=args.comparison_variant or None,
        backbone_seeds=args.backbone_seeds or [11, 17, 23],
        config=RetrievalComparisonConfig(
            mate_min_df=args.mate_min_df,
            max_vocab_size=args.max_vocab_size,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
    )
    print(f"Wrote retrieval comparison report to {result['md_path']}")


if __name__ == "__main__":
    main()
