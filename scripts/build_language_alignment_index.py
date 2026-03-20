"""Build split-safe language-sidecar alignment outputs for week-4."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.language_alignment import LanguageAlignmentConfig, build_language_alignment_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervised-train", required=True)
    parser.add_argument("--supervised-val", required=True)
    parser.add_argument("--supervised-test", required=True)
    parser.add_argument("--mate-path", required=True)
    parser.add_argument("--puzzle-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chessgpt-text-path", default=None)
    parser.add_argument("--chessgpt-conversation-path", default=None)
    parser.add_argument("--disable-fen4", action="store_true")
    parser.add_argument("--disable-puzzle-move-conditioned", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_language_alignment_index(
        supervised_train_path=args.supervised_train,
        supervised_val_path=args.supervised_val,
        supervised_test_path=args.supervised_test,
        mate_path=args.mate_path,
        puzzle_path=args.puzzle_path,
        output_root=args.output_root,
        chessgpt_text_path=args.chessgpt_text_path,
        chessgpt_conversation_path=args.chessgpt_conversation_path,
        config=LanguageAlignmentConfig(
            allow_fen_4field=not args.disable_fen4,
            allow_move_conditioned_for_puzzles=not args.disable_puzzle_move_conditioned,
        ),
    )
    print(f"Wrote alignment manifest to {result['manifest_path']}")


if __name__ == "__main__":
    main()
