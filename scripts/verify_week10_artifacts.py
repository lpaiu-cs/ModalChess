"""Verify week-10 annotated-sidecar and retrieval artifacts on disk."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.eval.week10_artifact_verification import verify_week10_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="outputs/week11/verification")
    parser.add_argument("--regenerate-if-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regenerate-if-stale", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_week10_artifacts(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        regenerate_if_missing=args.regenerate_if_missing,
        regenerate_if_stale=args.regenerate_if_stale,
    )
    print(f"Wrote verification report to {result['md_path']}")


if __name__ == "__main__":
    main()
