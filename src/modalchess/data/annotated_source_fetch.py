"""Local inventory and reproducible manifests for annotated comment sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import yaml

from modalchess.data.preprocessing_common import write_yaml


@dataclass(slots=True)
class AnnotatedSourceSpec:
    source_name: str
    source_family: str
    local_path: Path
    source_url: str | None
    repo_id: str | None
    version: str | None
    status: str
    usability: str
    note: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_size(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += int(file_path.stat().st_size)
    return total


def _waterhorse_version() -> str | None:
    prior_manifest = Path("data/pilot/language_probe_v4/manifests/aux_fetch_lock.yaml")
    if not prior_manifest.exists():
        return None
    payload = yaml.safe_load(prior_manifest.read_text(encoding="utf-8")) or {}
    artifacts = payload.get("artifacts") or []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("source_name") == "waterhorse_chess_data":
            return str(artifact.get("version") or artifact.get("checksum") or "")
    return None


def default_annotated_source_specs() -> list[AnnotatedSourceSpec]:
    waterhorse_version = _waterhorse_version()
    return [
        AnnotatedSourceSpec(
            source_name="waterhorse_annotated_pgn",
            source_family="annotated_pgn",
            local_path=Path("data/pilot/raw/hf/waterhorse_chess_data/chessgpt_data/annotated_pgn"),
            source_url="https://huggingface.co/datasets/Waterhorse/chess_data",
            repo_id="Waterhorse/chess_data",
            version=waterhorse_version,
            status="reused_local",
            usability="usable",
            note="Primary move-conditioned annotated PGN source.",
        ),
        AnnotatedSourceSpec(
            source_name="waterhorse_c4_manifest",
            source_family="text_corpus_manifest",
            local_path=Path("data/pilot/raw/hf/waterhorse_chess_data/chessgpt_data/c4"),
            source_url="https://huggingface.co/datasets/Waterhorse/chess_data",
            repo_id="Waterhorse/chess_data",
            version=waterhorse_version,
            status="reused_local",
            usability="audited_only",
            note="Dataset metadata only; not a move-conditioned comment source.",
        ),
        AnnotatedSourceSpec(
            source_name="waterhorse_ccrl_manifest",
            source_family="engine_corpus_manifest",
            local_path=Path("data/pilot/raw/hf/waterhorse_chess_data/chessgpt_data/ccrl"),
            source_url="https://huggingface.co/datasets/Waterhorse/chess_data",
            repo_id="Waterhorse/chess_data",
            version=waterhorse_version,
            status="reused_local",
            usability="audited_only",
            note="Dataset metadata only; not a comment-bearing move-conditioned source.",
        ),
        AnnotatedSourceSpec(
            source_name="waterhorse_chess_modeling_manifest",
            source_family="modeling_corpus_manifest",
            local_path=Path("data/pilot/raw/hf/waterhorse_chess_data/chessgpt_data/chess_modeling"),
            source_url="https://huggingface.co/datasets/Waterhorse/chess_data",
            repo_id="Waterhorse/chess_data",
            version=waterhorse_version,
            status="reused_local",
            usability="audited_only",
            note="Dataset metadata only; not a comment-bearing move-conditioned source.",
        ),
        AnnotatedSourceSpec(
            source_name="mate_strategy_zip",
            source_family="mate_strategy_pairwise",
            local_path=Path("data/pilot/raw/hf/mate_dataset/strategy.zip"),
            source_url=None,
            repo_id="mate_dataset",
            version="local",
            status="reused_local",
            usability="usable",
            note="Board-anchored pairwise strategy explanations with explicit FEN and chosen move.",
        ),
        AnnotatedSourceSpec(
            source_name="mate_tactic_zip",
            source_family="mate_tactic_pairwise",
            local_path=Path("data/pilot/raw/hf/mate_dataset/tactic.zip"),
            source_url=None,
            repo_id="mate_dataset",
            version="local",
            status="reused_local",
            usability="usable",
            note="Board-anchored pairwise tactic explanations with explicit FEN and chosen move.",
        ),
        AnnotatedSourceSpec(
            source_name="mate_both_zip",
            source_family="mate_both_pairwise",
            local_path=Path("data/pilot/raw/hf/mate_dataset/both.zip"),
            source_url=None,
            repo_id="mate_dataset",
            version="local",
            status="reused_local",
            usability="usable",
            note="Board-anchored pairwise strategy+tactic explanations with explicit FEN and chosen move.",
        ),
        AnnotatedSourceSpec(
            source_name="mate_testset_zip",
            source_family="mate_testset_pairwise",
            local_path=Path("data/pilot/raw/hf/mate_dataset/testset.zip"),
            source_url=None,
            repo_id="mate_dataset",
            version="local",
            status="reused_local",
            usability="usable",
            note="Held-out style pairwise explanation bundle with explicit FEN and chosen move.",
        ),
        AnnotatedSourceSpec(
            source_name="mate_no_explain_zip",
            source_family="mate_no_explain_pairwise",
            local_path=Path("data/pilot/raw/hf/mate_dataset/no_explain.zip"),
            source_url=None,
            repo_id="mate_dataset",
            version="local",
            status="reused_local",
            usability="skipped",
            note="Explicit FEN and chosen move exist, but comment text is absent by design.",
        ),
        AnnotatedSourceSpec(
            source_name="chessgpt_text_sample",
            source_family="sample_text_only",
            local_path=Path("data/pilot/samples/chessgpt_text_corpus.jsonl"),
            source_url=None,
            repo_id=None,
            version="local",
            status="reused_local",
            usability="skipped",
            note="Board-anchored sample, but not move-conditioned comment data.",
        ),
        AnnotatedSourceSpec(
            source_name="chessgpt_conversation_sample",
            source_family="sample_text_only",
            local_path=Path("data/pilot/samples/chessgpt_conversation_corpus.jsonl"),
            source_url=None,
            repo_id=None,
            version="local",
            status="reused_local",
            usability="skipped",
            note="Conversation sample only; not move-conditioned comment data.",
        ),
    ]


def fetch_more_annotated_sources(
    *,
    output_root: str | Path = "data/pilot/annotated_sidecar_v2_sources",
) -> dict[str, Any]:
    output_dir = Path(output_root)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    notes: list[str] = []
    for spec in default_annotated_source_specs():
        exists = spec.local_path.exists()
        file_size = _path_size(spec.local_path) if exists else None
        checksum = _sha256_file(spec.local_path) if exists and spec.local_path.is_file() else None
        artifact = {
            "source_name": spec.source_name,
            "source_family": spec.source_family,
            "source_url": spec.source_url,
            "repo_id": spec.repo_id,
            "version": spec.version,
            "local_path": str(spec.local_path),
            "file_size": file_size,
            "checksum": checksum,
            "fetch_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": spec.status if exists else "missing_local",
            "usability": spec.usability if exists else "missing",
            "note": spec.note,
        }
        artifacts.append(artifact)
        notes.append(
            f"- `{spec.source_name}` ({spec.source_family}): status={artifact['status']}, "
            f"usability={artifact['usability']}, local_path={spec.local_path}"
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifacts": artifacts,
    }
    lock_path = manifests_dir / "source_fetch_lock.yaml"
    notes_path = manifests_dir / "source_fetch_notes.md"
    write_yaml(lock_path, manifest)
    notes_path.write_text("# Annotated Source Fetch Notes\n\n" + "\n".join(notes) + "\n", encoding="utf-8")
    return {
        "lock_path": str(lock_path),
        "notes_path": str(notes_path),
        "manifest": manifest,
    }
