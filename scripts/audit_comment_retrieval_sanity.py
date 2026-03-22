"""Audit week-10/11 comment retrieval for stability and duplicate sensitivity."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.eval.comment_retrieval_sanity import (
    CommentRetrievalSanityConfig,
    audit_comment_retrieval_sanity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="outputs/week11/comment_retrieval_sanity")
    parser.add_argument("--alternate-salt", default="modalchess_week11_comment_eval_alt")
    parser.add_argument("--larger-subset-salt", default="modalchess_week11_comment_eval_large")
    parser.add_argument("--train-limit", type=int, default=100000)
    parser.add_argument("--val-limit", type=int, default=5000)
    parser.add_argument("--test-limit", type=int, default=5000)
    parser.add_argument("--larger-val-limit", type=int, default=10000)
    parser.add_argument("--larger-test-limit", type=int, default=10000)
    parser.add_argument("--mate-min-df", type=int, default=25)
    parser.add_argument("--max-vocab-size", type=int, default=512)
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_comment_retrieval_sanity(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        config=CommentRetrievalSanityConfig(
            alternate_salt=args.alternate_salt,
            larger_subset_salt=args.larger_subset_salt,
            train_limit=args.train_limit,
            val_limit=args.val_limit,
            test_limit=args.test_limit,
            larger_val_limit=args.larger_val_limit,
            larger_test_limit=args.larger_test_limit,
            mate_min_df=args.mate_min_df,
            max_vocab_size=args.max_vocab_size,
            embedding_batch_size=args.embedding_batch_size,
        ),
    )
    print(f"Wrote comment retrieval sanity report to {result['md_path']}")


if __name__ == "__main__":
    main()
