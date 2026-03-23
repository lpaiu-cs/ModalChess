"""Build the week-18 winner-stability report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.eval.winner_stability import build_winner_stability_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--week17-results",
        default="outputs/week17/comment_retrieval_v6/comment_retrieval_results.json",
    )
    parser.add_argument(
        "--week18-holdout-results",
        default="outputs/week18/source_holdout_balanced/results.json",
    )
    parser.add_argument("--output-dir", default="outputs/week18/winner_stability")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_winner_stability_report(
        week17_results_path=args.week17_results,
        week18_holdout_path=args.week18_holdout_results,
        output_dir=args.output_dir,
    )
    print(f"Wrote winner stability report to {result['md_path']}")


if __name__ == "__main__":
    main()
