"""Waterhorse/chess_data를 schema별 normalized corpora로 분리한다."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.chessgpt_normalizer import (
    ChessGptNormalizationConfig,
    write_normalized_chessgpt_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="입력 Waterhorse corpus 파일 경로")
    parser.add_argument("--output-dir", required=True, help="출력 디렉터리")
    parser.add_argument("--source-version", default=None, help="원천 버전 문자열")
    parser.add_argument("--source-date", default=None, help="원천 날짜 문자열")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ChessGptNormalizationConfig(
        source_version=args.source_version,
        source_date=args.source_date,
    )
    write_normalized_chessgpt_corpus(args.inputs, args.output_dir, config)
    print(f"Wrote normalized ChessGPT corpora to {args.output_dir}")


if __name__ == "__main__":
    main()
