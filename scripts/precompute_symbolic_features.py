"""corpus 3 split의 심볼릭 특징(board move + text mention)을 precompute."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.align.symbolic_features import precompute_symbolic_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", default="outputs/scale_v1/gate2_comment/corpus")
    parser.add_argument("--output-root", default="outputs/connector_hybrid_v1/symbolic_features")
    parser.add_argument("--family", default="annotated_sidecar")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_root = Path(args.corpus_root)
    output_root = Path(args.output_root)
    for split in ("train", "val", "test"):
        corpus_path = corpus_root / f"{args.family}_{split}.jsonl"
        output_path = output_root / f"{args.family}_{split}_features.pt"
        info = precompute_symbolic_features(corpus_path, output_path)
        print(f"{split}: n={info['n_rows']} board_dim={info['board_dim']} "
              f"text_dim={info['text_dim']} -> {info['output']}")


if __name__ == "__main__":
    main()
