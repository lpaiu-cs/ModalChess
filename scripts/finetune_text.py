"""Text encoder fine-tune CLI (레버 ②b): 학습 후 test 채점(null + 세그먼트)까지 실행."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.align.finetune_text import evaluate_finetuned, finetune_text_encoder  # noqa: E402
from modalchess.utils.config import load_yaml_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-board", default=None)
    parser.add_argument("--val-board", default=None)
    parser.add_argument("--test-board", default=None)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    for cli_key, config_key in (
        ("seed", "seed"), ("output_dir", "output_dir"),
        ("train_board", "train_board"), ("val_board", "val_board"), ("test_board", "test_board"),
    ):
        value = getattr(args, cli_key)
        if value is not None:
            config[config_key] = value

    if not args.eval_only:
        summary = finetune_text_encoder(config)
        print(f"best_epoch={summary['best_epoch']} val_selection={summary['best_val_selection_score']:.5f}")

    eval_config = dict(config)
    eval_config["finetuned"] = str(Path(config["output_dir"]) / "finetuned_text.pt")
    result = evaluate_finetuned(eval_config)
    real = result["real"]
    seg = result["segment"]
    v = result["verdict"]
    print(f"n_test={result['n_test']}  t2b_mrr={real['text_to_board']['mrr']:.5f}  "
          f"b2t_mrr={real['board_to_text']['mrr']:.5f}  t2b R@50={real['text_to_board']['r@50']:.4f}")
    print(f"segment n={seg['n_segment']}  t2b mrr={seg['text_to_board']['mrr']:.5f} "
          f"R@10={seg['text_to_board']['r@10']:.4f} R@50={seg['text_to_board']['r@50']:.4f}")
    print(f"segment gain vs hybrid baseline: {v['segment_gain_vs_hybrid']:.2f}x "
          f"(min-bar 1.3x: {v['passes_segment_min_bar']})")
    print(f"above within-family null: t2b={v['t2b_above_within_family_null']} "
          f"b2t={v['b2t_above_within_family_null']}")


if __name__ == "__main__":
    main()
