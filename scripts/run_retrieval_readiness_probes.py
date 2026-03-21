"""Run retrieval-style frozen-backbone language readiness probes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.eval.language_retrieval import run_retrieval_readiness_probes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone-seed", type=int, action="append", dest="backbone_seeds", default=[])
    parser.add_argument("--mate-min-train-positive", type=int, default=25)
    parser.add_argument("--puzzle-min-train-positive", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_retrieval_readiness_probes(
        embedding_root=args.embedding_root,
        corpus_root=args.corpus_root,
        target_root=args.target_root,
        output_dir=args.output_dir,
        backbone_seeds=args.backbone_seeds or [11],
        mate_min_train_positive=args.mate_min_train_positive,
        puzzle_min_train_positive=args.puzzle_min_train_positive,
    )
    print(f"Wrote retrieval probe summary to {result['summary_path']}")


if __name__ == "__main__":
    main()
