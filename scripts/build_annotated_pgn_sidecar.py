"""Build a move-conditioned comment sidecar from Waterhorse annotated PGN."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.annotated_pgn_sidecar import (
    AnnotatedPgnSidecarConfig,
    build_annotated_pgn_sidecar,
    write_annotated_sidecar_report,
)
from modalchess.data.preprocessing_common import StableSplitConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        default="data/pilot/raw/hf/waterhorse_chess_data/chessgpt_data/annotated_pgn",
    )
    parser.add_argument("--output-root", default="data/pilot/annotated_sidecar_v1")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-salt", default="modalchess_week9_annotated_sidecar")
    parser.add_argument(
        "--include-history-fens",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--compare-aux-root", default="data/pilot/language_probe_v4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_annotated_pgn_sidecar(
        input_root=args.input_root,
        output_root=args.output_root,
        config=AnnotatedPgnSidecarConfig(
            split_config=StableSplitConfig(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                salt=args.split_salt,
            ),
            include_history_fens=args.include_history_fens,
        ),
    )
    report_result = write_annotated_sidecar_report(
        input_root=args.output_root,
        output_dir=Path(args.output_root) / "reports",
        compare_aux_root=args.compare_aux_root,
    )
    print(f"Wrote annotated sidecar manifest to {result['manifest_path']}")
    print(f"Wrote annotated sidecar report to {report_result['report_json']}")


if __name__ == "__main__":
    main()
