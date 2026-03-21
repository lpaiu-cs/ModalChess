"""Fetch or snapshot auxiliary language sources for week-8."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.aux_source_fetch import (
    record_local_aux_source,
    record_unavailable_aux_source,
    snapshot_aux_hf_dataset,
    write_aux_fetch_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/pilot/raw/hf")
    parser.add_argument(
        "--manifest-path",
        default="data/pilot/language_probe_v4/manifests/aux_fetch_lock.yaml",
    )
    parser.add_argument(
        "--notes-path",
        default="data/pilot/language_probe_v4/manifests/aux_fetch_notes.md",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include-waterhorse",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    entries = []

    if args.include_waterhorse:
        try:
            entries.append(
                snapshot_aux_hf_dataset(
                    source_name="waterhorse_chess_data",
                    repo_id="Waterhorse/chess_data",
                    destination=raw_root / "waterhorse_chess_data",
                    allow_patterns=(
                        "README.md",
                        "chessgpt_data/annotated_pgn/*",
                        "chessgpt_data/c4/dataset_info.json",
                        "chessgpt_data/ccrl/dataset_info.json",
                        "chessgpt_data/chess_modeling/dataset_info.json",
                    ),
                    force=args.force,
                    usability="partial",
                    note="Bounded week-8 snapshot: README + annotated_pgn shards + selected dataset_info files only.",
                )
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            entries.append(
                record_unavailable_aux_source(
                    source_name="waterhorse_chess_data",
                    source_url="https://huggingface.co/datasets/Waterhorse/chess_data",
                    repo_id="Waterhorse/chess_data",
                    note=f"Snapshot failed: {exc}",
                )
            )

    entries.append(
        record_local_aux_source(
            source_name="chessgpt_text_sample",
            path="data/pilot/samples/chessgpt_text_corpus.jsonl",
            source_url="data/pilot/samples/chessgpt_text_corpus.jsonl",
            usability="usable",
            note="Previously normalized local sample corpus.",
        )
    )
    entries.append(
        record_local_aux_source(
            source_name="chessgpt_conversation_sample",
            path="data/pilot/samples/chessgpt_conversation_corpus.jsonl",
            source_url="data/pilot/samples/chessgpt_conversation_corpus.jsonl",
            usability="usable",
            note="Previously normalized local sample corpus.",
        )
    )

    write_aux_fetch_lock(
        entries=entries,
        manifest_path=args.manifest_path,
        notes_path=args.notes_path,
    )
    print(f"Wrote auxiliary fetch lock to {args.manifest_path}")


if __name__ == "__main__":
    main()
