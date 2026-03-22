"""Build a broader move-conditioned comment sidecar from multiple audited sources."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.multisource_annotated_sidecar import (
    MultisourceAnnotatedSidecarConfig,
    build_multisource_annotated_sidecar,
)
from modalchess.data.preprocessing_common import StableSplitConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waterhorse-input-root", default="data/pilot/annotated_sidecar_v1")
    parser.add_argument("--output-root", default="data/pilot/annotated_sidecar_v4_multisource")
    parser.add_argument("--train-family-cap", type=int, default=30000)
    parser.add_argument("--val-family-cap", type=int, default=4000)
    parser.add_argument("--test-family-cap", type=int, default=4000)
    parser.add_argument("--train-min-family", type=int, default=200)
    parser.add_argument("--val-min-family", type=int, default=25)
    parser.add_argument("--test-min-family", type=int, default=25)
    parser.add_argument("--salt", default="modalchess_week15_multisource")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MultisourceAnnotatedSidecarConfig(
        split_config=StableSplitConfig(salt=args.salt),
        source_family_caps={
            "train": args.train_family_cap,
            "val": args.val_family_cap,
            "test": args.test_family_cap,
        },
        min_source_family_presence={
            "train": args.train_min_family,
            "val": args.val_min_family,
            "test": args.test_min_family,
        },
    )
    result = build_multisource_annotated_sidecar(
        waterhorse_input_root=args.waterhorse_input_root,
        output_root=args.output_root,
        config=config,
    )
    print(f"Wrote multisource sidecar manifest to {result['manifest_path']}")


if __name__ == "__main__":
    main()
