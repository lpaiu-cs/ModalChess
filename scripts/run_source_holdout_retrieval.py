"""Run source-holdout and source-type retrieval evaluation on frozen embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.eval.source_holdout_retrieval import run_source_holdout_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-root", default="data/pilot/annotated_sidecar_holdout_v1")
    parser.add_argument("--full-embedding-root", default="outputs/week16/full_eval_embeddings")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--backbone-seed", type=int, action="append", dest="backbone_seeds", default=[])
    parser.add_argument("--mate-min-df", type=int, default=25)
    parser.add_argument("--max-vocab-size", type=int, default=512)
    parser.add_argument("--probe-model", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_source_holdout_retrieval(
        holdout_root=args.holdout_root,
        full_embedding_root=args.full_embedding_root,
        output_dir=args.output_dir,
        categories=set(args.category or ["mixed_baseline", "coarse_source_holdout", "source_family_holdout"]),
        backbone_seeds=args.backbone_seeds or [11, 17, 23],
        mate_min_df=args.mate_min_df,
        max_vocab_size=args.max_vocab_size,
        probe_models=args.probe_model or ["linear"],
    )
    print(f"Wrote source holdout retrieval summary to {result['summary_path']}")


if __name__ == "__main__":
    main()
