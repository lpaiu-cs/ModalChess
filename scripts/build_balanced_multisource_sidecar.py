"""Build balanced and style-normalized multisource annotated comment variants."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.balanced_multisource_sidecar import (
    BalancedMultisourceConfig,
    build_balanced_multisource_sidecar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/pilot/annotated_sidecar_v4_multisource")
    parser.add_argument("--output-root", default="data/pilot/annotated_sidecar_v5_balanced")
    parser.add_argument("--train-family-cap", type=int, default=5000)
    parser.add_argument("--val-family-cap", type=int, default=700)
    parser.add_argument("--test-family-cap", type=int, default=700)
    parser.add_argument("--train-source-type-cap", type=int, default=25000)
    parser.add_argument("--val-source-type-cap", type=int, default=3500)
    parser.add_argument("--test-source-type-cap", type=int, default=3500)
    parser.add_argument("--salt", default="modalchess_week17_balanced")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_balanced_multisource_sidecar(
        input_root=args.input_root,
        output_root=args.output_root,
        config=BalancedMultisourceConfig(
            family_caps={
                "train": args.train_family_cap,
                "val": args.val_family_cap,
                "test": args.test_family_cap,
            },
            source_type_caps={
                "train": args.train_source_type_cap,
                "val": args.val_source_type_cap,
                "test": args.test_source_type_cap,
            },
            salt=args.salt,
        ),
    )
    print(f"Wrote balanced multisource manifest to {result['manifest_path']}")


if __name__ == "__main__":
    main()
