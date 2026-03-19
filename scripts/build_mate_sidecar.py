"""MATE 데이터셋을 language sidecar JSONL로 변환한다."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.mate_sidecar import MateSidecarBuildConfig, write_mate_sidecar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="입력 MATE 원천 파일 경로")
    parser.add_argument("--output-path", required=True, help="출력 JSONL 경로")
    parser.add_argument("--source-version", default=None, help="원천 버전 문자열")
    parser.add_argument("--source-date", default=None, help="원천 날짜 문자열")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MateSidecarBuildConfig(
        source_version=args.source_version,
        source_date=args.source_date,
    )
    write_mate_sidecar(args.inputs, args.output_path, config)
    print(f"Wrote MATE sidecar to {args.output_path}")


if __name__ == "__main__":
    main()
