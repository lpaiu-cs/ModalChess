"""Connector 평가 CLI (strict R@k + global/within-family null)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.align.eval_connector import evaluate_connector  # noqa: E402
from modalchess.utils.config import load_yaml_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--connector", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.connector is not None:
        config["connector"] = args.connector
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    result = evaluate_connector(config)
    v = result["verdict"]
    real = result["real"]
    print(f"n_test={result['n_test']}  t2b_mrr={v['t2b_mrr']:.5f}  b2t_mrr={v['b2t_mrr']:.5f}")
    print(f"  t2b R@10={real['text_to_board']['r@10']:.4f} R@50={real['text_to_board']['r@50']:.4f}")
    print(f"  vs frozen-probe 0.01084: {v['t2b_over_frozen_probe']:.2f}x (min-bar 1.3x: {v['beats_frozen_probe_min_bar']})")
    print(f"  above global null: {v['t2b_above_global_null']}  "
          f"above within-family null: {v['t2b_above_within_family_null']}  "
          f"b2t above within-family null: {v['b2t_above_within_family_null']}")


if __name__ == "__main__":
    main()
