"""Run week-18 readiness falsification analyses on balanced retrieval variants."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.eval.readiness_falsification import (  # noqa: E402
    ReadinessFalsificationConfig,
    run_readiness_falsification,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/week18")
    parser.add_argument("--checkpoint-eval-root", default="outputs/week18/checkpoint_honesty")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1234)
    parser.add_argument("--null-seed", type=int, default=20260324)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_readiness_falsification(
        ReadinessFalsificationConfig(
            output_root=args.output_root,
            checkpoint_eval_root=args.checkpoint_eval_root,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            null_seed=args.null_seed,
        )
    )
    print(f"Wrote readiness falsification outputs to {args.output_root}")
    print(f"Git commit: {result['git_commit']}")


if __name__ == "__main__":
    main()
