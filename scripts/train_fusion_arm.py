"""P1 arm 학습/평가 CLI: 학습(해당 시) 후 test 채점 + shuffled-board null까지 한 번에."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.fusion.fusion_run import run_arm  # noqa: E402
from modalchess.utils.config import load_yaml_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True,
                        choices=["board", "rawboard", "blind", "fen_soft", "fen_zs"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for cli_key, cfg_key in (
        ("seed", "seed"), ("output_dir", "output_dir"), ("epochs", "epochs"),
        ("limit_train", "limit_train"), ("limit_val", "limit_val"), ("limit_test", "limit_test"),
    ):
        value = getattr(args, cli_key)
        if value is not None:
            config[cfg_key] = value
    run_arm(config, args.arm)


if __name__ == "__main__":
    main()
