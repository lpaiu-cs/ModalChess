"""Build board-anchored/text-only auxiliary language corpora for week-7."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.aux_language import AuxLanguageBuildConfig, build_aux_language_corpora
from modalchess.data.preprocessing_common import StableSplitConfig


def _parse_source_args(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--source must use name=path form: {value}")
        name, path = value.split("=", 1)
        parsed[name] = path
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/pilot/language_probe_v3")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-salt", default="modalchess_week7_aux")
    parser.add_argument("--source", action="append", default=[], help="name=path form; repeatable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_aux_language_corpora(
        source_paths=_parse_source_args(args.source) if args.source else None,
        output_root=args.output_root,
        config=AuxLanguageBuildConfig(
            split_config=StableSplitConfig(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                salt=args.split_salt,
            )
        ),
    )
    print(f"Wrote auxiliary source manifest to {result['manifest_path']}")


if __name__ == "__main__":
    main()
