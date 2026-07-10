"""Oracle ceiling 진단 CLI: pair 데이터의 retrieval 상한/하한을 심볼릭하게 잰다."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.align.oracle_ceiling import KS, run_oracle_ceiling  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        default="outputs/scale_v1/gate2_comment/corpus/annotated_sidecar_test.jsonl",
    )
    parser.add_argument(
        "--test-board",
        default="outputs/scale_v1/gate2_comment/embedding_exports/g3/seed11/annotated_sidecar_test_embeddings.pt",
    )
    parser.add_argument(
        "--test-text",
        default="outputs/connector_v1/text_embeddings/annotated_sidecar_test_text.pt",
    )
    parser.add_argument("--pool", default="board_pooled")
    parser.add_argument("--output-dir", default="outputs/connector_v1/oracle_ceiling")
    return parser.parse_args()


def _fmt(metrics: dict[str, float]) -> str:
    cells = [f"mrr={metrics['mrr']:.5f}"]
    cells += [f"r@{k}={metrics[f'r@{k}']:.4f}" for k in KS]
    return "  ".join(cells)


def main() -> None:
    args = parse_args()
    report = run_oracle_ceiling(
        {
            "corpus": args.corpus,
            "test_board": args.test_board,
            "test_text": args.test_text,
            "pool": args.pool,
            "output_dir": args.output_dir,
        }
    )
    print(f"pool n={report['n_pool']}  invalid_moves={report['n_invalid_moves']}")
    print(f"move_conditioned_fraction={report['move_conditioned_fraction']:.4f}")
    print("\n[duplicate ceiling] (어떤 함수도 못 넘는 상한)")
    for direction in ("text_to_board", "board_to_text"):
        entry = report["duplicate_ceiling"][direction]
        print(f"  {direction}:")
        for kind in ("assignment", "tie", "combined_min"):
            print(f"    {kind:<12} {_fmt(entry[kind])}")
    print("\n[symbolic oracle] (코멘트가 속성을 완벽 전달한다고 가정한 상한)")
    for variant, entry in report["symbolic_oracle"].items():
        print(f"  {variant}:")
        for direction in ("text_to_board", "board_to_text"):
            print(f"    {direction:<14} {_fmt(entry[direction])}")
    print("\n[mention baseline] (SAN/UCI 문자열 매칭 구성적 하한)")
    for direction in ("text_to_board", "board_to_text"):
        print(f"    {direction:<14} {_fmt(report['mention_baseline'][direction])}")
    print("\n[move-conditioned fraction by family]")
    for family, entry in report["move_conditioned_by_family"].items():
        print(f"    {family:<32} n={entry['n']:<5} {entry['fraction']:.4f}")


if __name__ == "__main__":
    main()
