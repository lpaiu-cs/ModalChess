"""Lichess puzzle 원천을 ModalChess puzzle sidecar JSONL로 변환한다."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.preprocessing_common import StableSplitConfig
from modalchess.data.puzzle_sidecar import PuzzleSidecarBuildConfig, write_puzzle_sidecar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="입력 puzzle CSV/JSONL 경로")
    parser.add_argument("--output-path", required=True, help="출력 JSONL 경로")
    parser.add_argument("--source-version", default=None, help="원천 버전 문자열")
    parser.add_argument("--source-date", default=None, help="원천 날짜 문자열")
    parser.add_argument("--assign-split", action="store_true", help="game_id 기준 stable split 부여")
    parser.add_argument("--emit-legal-moves", action="store_true", help="debug용 legal_moves_uci를 기록")
    parser.add_argument("--max-rows", type=int, default=None, help="최대 유지 row 수")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="train split 비율")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="val split 비율")
    parser.add_argument("--split-salt", default="modalchess", help="stable split salt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PuzzleSidecarBuildConfig(
        source_version=args.source_version,
        source_date=args.source_date,
        emit_legal_moves=args.emit_legal_moves,
        assign_split=args.assign_split,
        max_rows=args.max_rows,
        split_config=StableSplitConfig(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            salt=args.split_salt,
        ),
    )
    write_puzzle_sidecar(args.inputs, args.output_path, config)
    print(f"Wrote puzzle sidecar to {args.output_path}")


if __name__ == "__main__":
    main()
