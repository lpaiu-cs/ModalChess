"""Build standalone week-5/week-6 language probe corpora."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.preprocessing_common import StableSplitConfig
from modalchess.data.probe_corpora import ProbeCorpusConfig, build_probe_corpora
from modalchess.data.probe_reports import write_probe_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mate-path", required=True)
    parser.add_argument("--puzzle-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-salt", default="modalchess_week5_probe")
    parser.add_argument(
        "--prefer-game-id-group-split",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-game-id-group-size", type=int, default=2)
    parser.add_argument("--report-output-dir", default=None)
    parser.add_argument("--compare-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_output_dir = args.report_output_dir or str(Path(args.output_root) / "reports")
    result = build_probe_corpora(
        mate_path=args.mate_path,
        puzzle_path=args.puzzle_path,
        output_root=args.output_root,
        config=ProbeCorpusConfig(
            split_config=StableSplitConfig(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                salt=args.split_salt,
            ),
            prefer_game_id_group_split=args.prefer_game_id_group_split,
            min_game_id_group_size=args.min_game_id_group_size,
        ),
    )
    write_probe_reports(
        input_root=args.output_root,
        output_dir=report_output_dir,
        compare_root=args.compare_root,
    )
    print(f"Wrote probe manifest to {result['manifest_path']}")


if __name__ == "__main__":
    main()
