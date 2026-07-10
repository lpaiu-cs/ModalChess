"""Connector 학습 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.align.train_connector import train_connector  # noqa: E402
from modalchess.utils.config import load_yaml_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--projection", default=None)
    parser.add_argument("--balance", default=None)
    parser.add_argument("--pool", default=None)
    parser.add_argument("--train-board", default=None)
    parser.add_argument("--val-board", default=None)
    parser.add_argument("--test-board", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    for key in ("seed", "output_dir", "projection", "balance", "pool"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    for cli_key, config_key in (("train_board", "train_board"), ("val_board", "val_board"), ("test_board", "test_board")):
        value = getattr(args, cli_key)
        if value is not None:
            config[config_key] = value
    summary = train_connector(config)
    print(f"best_epoch={summary['best_epoch']} val_selection={summary['best_val_selection_score']:.5f}")
    print(f"val t2b={summary['best_val_metrics']['text_to_board']['mrr']:.5f} "
          f"b2t={summary['best_val_metrics']['board_to_text']['mrr']:.5f}")


if __name__ == "__main__":
    main()
