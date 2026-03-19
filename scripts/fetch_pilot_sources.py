"""실제 파일럿용 raw source를 fetch/snapshot하고 lock manifest를 남긴다."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.data.source_fetch import (
    FetchedArtifact,
    LICHESS_EVAL_URL,
    LICHESS_PUZZLE_URL,
    download_url_artifact,
    inspect_url_metadata,
    lookup_lichess_standard_checksum,
    record_snapshot_only_entry,
    resolve_lichess_standard_url,
    snapshot_hf_dataset,
    write_fetch_lock_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/pilot/raw", help="raw source 저장 디렉터리")
    parser.add_argument(
        "--manifest-path",
        default="data/pilot/manifests/raw_fetch_lock.yaml",
        help="fetch lock manifest 경로",
    )
    parser.add_argument(
        "--notes-path",
        default="data/pilot/manifests/raw_fetch_notes.md",
        help="human-readable fetch note 경로",
    )
    parser.add_argument(
        "--standard-version",
        default="2015-01",
        help="Lichess standard month version 또는 latest",
    )
    parser.add_argument("--force", action="store_true", help="기존 artifact가 있어도 다시 받는다")
    parser.add_argument(
        "--materialize-eval",
        action="store_true",
        help="거대한 lichess eval artifact도 실제 다운로드한다",
    )
    parser.add_argument(
        "--include-waterhorse",
        action="store_true",
        help="보조 Waterhorse snapshot도 함께 받는다",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    entries: list[FetchedArtifact] = []
    puzzle_last_modified = inspect_url_metadata(LICHESS_PUZZLE_URL).get("Last-Modified", "unknown")
    eval_last_modified = inspect_url_metadata(LICHESS_EVAL_URL).get("Last-Modified", "unknown")

    standard_url, standard_version = resolve_lichess_standard_url(args.standard_version)
    standard_filename = standard_url.rstrip("/").split("/")[-1]
    entries.append(
        download_url_artifact(
            source_name="lichess_standard_pgn",
            url=standard_url,
            version=standard_version,
            destination=raw_root / "lichess" / "standard" / standard_filename,
            force=args.force,
            checksum_hint=lookup_lichess_standard_checksum(standard_version),
        )
    )

    entries.append(
        download_url_artifact(
            source_name="lichess_puzzle",
            url=LICHESS_PUZZLE_URL,
            version=puzzle_last_modified,
            destination=raw_root / "lichess" / "puzzles" / "lichess_db_puzzle.csv.zst",
            force=args.force,
        )
    )

    if args.materialize_eval:
        entries.append(
            download_url_artifact(
                source_name="lichess_eval",
                url=LICHESS_EVAL_URL,
                version=eval_last_modified,
                destination=raw_root / "lichess" / "evals" / "lichess_db_eval.jsonl.zst",
                force=args.force,
            )
        )
    else:
        entries.append(
            record_snapshot_only_entry(
                source_name="lichess_eval",
                url=LICHESS_EVAL_URL,
                version=eval_last_modified,
                note="Artifact is ~19.8 GB. Kept as snapshot-only by default for bounded pilot builds; pass --materialize-eval to download.",
            )
        )

    entries.append(
        snapshot_hf_dataset(
            source_name="mate_dataset",
            repo_id="OutFlankShu/MATE_DATASET",
            destination=raw_root / "hf" / "mate_dataset",
            force=args.force,
            allow_patterns=("README.md", "*.zip"),
        )
    )

    if args.include_waterhorse:
        entries.append(
            snapshot_hf_dataset(
                source_name="waterhorse_chess_data",
                repo_id="Waterhorse/chess_data",
                destination=raw_root / "hf" / "waterhorse_chess_data",
                force=args.force,
                allow_patterns=("README.md", "chessgpt_data/annotated_pgn/*"),
            )
        )

    write_fetch_lock_manifest(
        entries=entries,
        manifest_path=args.manifest_path,
        notes_path=args.notes_path,
    )
    print(f"Wrote fetch lock manifest to {args.manifest_path}")


if __name__ == "__main__":
    main()
