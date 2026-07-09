"""Comment corpus의 문장 임베딩을 1회 precompute (frozen MiniLM)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.align.text_embed import precompute_corpus_text_embeddings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", default="outputs/scale_v1/gate2_comment/corpus")
    parser.add_argument("--output-root", default="outputs/connector_v1/text_embeddings")
    parser.add_argument("--family", default="annotated_sidecar")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = precompute_corpus_text_embeddings(
        corpus_root=args.corpus_root,
        output_root=args.output_root,
        family=args.family,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    for split, path in written.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
