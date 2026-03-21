"""Export frozen-backbone embeddings for week-4 sidecar datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.eval.embedding_export import EmbeddingExportConfig, export_embeddings_for_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", action="append", default=[], help="name=path form; repeatable")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--include-square-tokens", action="store_true")
    parser.add_argument("--format", choices=("jsonl", "pt"), default="jsonl")
    return parser.parse_args()


def _parse_dataset_args(values: list[str]) -> dict[str, str]:
    datasets: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--dataset must use name=path form: {value}")
        name, path = value.split("=", 1)
        datasets[name] = path
    return datasets


def main() -> None:
    args = parse_args()
    result = export_embeddings_for_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_paths=_parse_dataset_args(args.dataset),
        output_dir=args.output_dir,
        config=EmbeddingExportConfig(
            batch_size=args.batch_size,
            include_square_tokens=args.include_square_tokens,
            output_format=args.format,
        ),
    )
    print(f"Wrote embedding manifest to {result['manifest_path']}")


if __name__ == "__main__":
    main()
