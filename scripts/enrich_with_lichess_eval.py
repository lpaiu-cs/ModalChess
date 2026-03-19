"""supervised JSONL에 Lichess evaluation 정보를 조인한다."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.eval_enrichment import EvalEnrichmentConfig, write_enriched_supervised_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervised", nargs="+", required=True, help="입력 supervised JSONL 경로")
    parser.add_argument("--evals", nargs="*", default=[], help="입력 evaluation JSONL/CSV 경로")
    parser.add_argument("--output-path", required=True, help="출력 JSONL 경로")
    parser.add_argument("--source-version", default=None, help="원천 버전 문자열")
    parser.add_argument("--source-date", default=None, help="원천 날짜 문자열")
    parser.add_argument(
        "--mate-mode",
        choices=("separate_field", "sentinel"),
        default="separate_field",
        help="mate 값을 별도 필드로 둘지 sentinel cp로 투영할지",
    )
    parser.add_argument("--mate-cp-sentinel", type=int, default=100000, help="mate sentinel cp 값")
    parser.add_argument("--drop-unmatched", action="store_true", help="eval 미일치 row를 버린다")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvalEnrichmentConfig(
        source_version=args.source_version,
        source_date=args.source_date,
        mate_mode=args.mate_mode,
        mate_cp_sentinel=args.mate_cp_sentinel,
        keep_unmatched_rows=not args.drop_unmatched,
    )
    write_enriched_supervised_jsonl(args.supervised, args.evals, args.output_path, config)
    print(f"Wrote enriched supervised JSONL to {args.output_path}")


if __name__ == "__main__":
    main()
