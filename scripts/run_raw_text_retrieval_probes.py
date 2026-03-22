"""Run week-7 raw-text and synthetic-tag retrieval probes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.eval.raw_text_retrieval import run_raw_text_retrieval_probes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", default="outputs/week6/embedding_exports")
    parser.add_argument("--corpus-root", default="data/pilot/language_probe_v2")
    parser.add_argument("--output-dir", default="outputs/week7/raw_text_retrieval")
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--backbone-seed", type=int, action="append", dest="backbone_seeds", default=[])
    parser.add_argument("--mate-min-df", type=int, default=50)
    parser.add_argument("--puzzle-min-df", type=int, default=25)
    parser.add_argument("--max-vocab-size", type=int, default=256)
    parser.add_argument("--probe-model", action="append", default=[])
    parser.add_argument("--output-prefix", default="raw_text_retrieval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_raw_text_retrieval_probes(
        embedding_root=args.embedding_root,
        corpus_root=args.corpus_root,
        output_dir=args.output_dir,
        backbone_seeds=args.backbone_seeds or [11, 17, 23],
        mate_min_df=args.mate_min_df,
        puzzle_min_df=args.puzzle_min_df,
        max_vocab_size=args.max_vocab_size,
        families=args.family or None,
        probe_models=args.probe_model or None,
        output_prefix=args.output_prefix,
    )
    print(f"Wrote raw-text retrieval summary to {result['summary_path']}")


if __name__ == "__main__":
    main()
