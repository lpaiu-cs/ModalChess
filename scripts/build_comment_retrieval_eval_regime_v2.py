"""Build a strict retrieval regime on top of the dedup-aware annotated corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.comment_retrieval_eval import CommentRetrievalEvalConfig
from modalchess.data.dedup_comment_eval import build_comment_retrieval_eval_regime_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/pilot/annotated_sidecar_eval_v2")
    parser.add_argument("--output-root", default="outputs/week12/comment_retrieval_v2")
    parser.add_argument("--train-limit", type=int, default=100000)
    parser.add_argument("--val-limit", type=int, default=5000)
    parser.add_argument("--test-limit", type=int, default=5000)
    parser.add_argument("--salt", default="modalchess_week12_comment_eval")
    parser.add_argument("--require-non-empty-comment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stratify-by", default="comment_source")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_comment_retrieval_eval_regime_v2(
        input_root=args.input_root,
        output_root=args.output_root,
        config=CommentRetrievalEvalConfig(
            train_limit=args.train_limit,
            val_limit=args.val_limit,
            test_limit=args.test_limit,
            salt=args.salt,
            require_non_empty_comment=args.require_non_empty_comment,
            stratify_by=args.stratify_by,
        ),
    )
    print(f"Wrote retrieval eval manifest to {result['manifest_path']}")


if __name__ == "__main__":
    main()
