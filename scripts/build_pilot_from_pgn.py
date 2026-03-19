"""PGN을 ModalChess supervised pilot JSONL로 변환한다."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.pgn_pilot import PgnPilotBuildConfig, write_supervised_pilot_from_pgn
from modalchess.data.preprocessing_common import StableSplitConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="입력 PGN 파일 경로")
    parser.add_argument("--output-dir", required=True, help="출력 디렉터리")
    parser.add_argument("--source-version", default=None, help="원천 버전 문자열")
    parser.add_argument("--source-date", default=None, help="원천 날짜 문자열")
    parser.add_argument("--rated-only", action="store_true", help="rated game만 유지")
    parser.add_argument("--allow-variants", action="store_true", help="variant game도 허용")
    parser.add_argument("--emit-legal-moves", action="store_true", help="debug용 legal_moves_uci를 기록")
    parser.add_argument("--min-game-plies", type=int, default=1, help="최소 game ply 수")
    parser.add_argument("--max-game-plies", type=int, default=None, help="최대 game ply 수")
    parser.add_argument("--min-ply-index", type=int, default=0, help="포지션 추출 최소 ply index")
    parser.add_argument("--max-ply-index", type=int, default=None, help="포지션 추출 최대 ply index")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="train split 비율")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="val split 비율")
    parser.add_argument("--split-salt", default="modalchess", help="stable split salt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PgnPilotBuildConfig(
        source_version=args.source_version,
        source_date=args.source_date,
        standard_only=not args.allow_variants,
        rated_only=args.rated_only,
        emit_legal_moves=args.emit_legal_moves,
        min_game_plies=args.min_game_plies,
        max_game_plies=args.max_game_plies,
        min_ply_index=args.min_ply_index,
        max_ply_index=args.max_ply_index,
        split_config=StableSplitConfig(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            salt=args.split_salt,
        ),
    )
    manifest = write_supervised_pilot_from_pgn(args.inputs, args.output_dir, config)
    print(f"Wrote supervised pilot splits to {args.output_dir}")
    print(f"Manifest: {ROOT / Path(manifest['outputs']['train']).parent / 'pgn_manifest.yaml'}")


if __name__ == "__main__":
    main()
