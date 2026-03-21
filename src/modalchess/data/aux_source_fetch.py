"""Auxiliary language source fetch helpers for week-8."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from modalchess.data.preprocessing_common import compute_file_sha256, write_yaml


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _directory_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


@dataclass(slots=True)
class AuxFetchArtifact:
    """Single auxiliary-source fetch lock entry."""

    source_name: str
    source_url: str | None
    repo_id: str | None
    version: str
    local_path: str | None
    file_size: int | None
    checksum: str | None
    fetch_timestamp: str
    status: str
    usability: str
    note: str | None = None


def snapshot_aux_hf_dataset(
    *,
    source_name: str,
    repo_id: str,
    destination: str | Path,
    allow_patterns: Iterable[str] | None = None,
    force: bool = False,
    usability: str = "partial",
    note: str | None = None,
) -> AuxFetchArtifact:
    """Snapshot a Hugging Face dataset repo for auxiliary week-8 use."""
    api = HfApi()
    info = api.dataset_info(repo_id)
    destination_path = Path(destination)
    fetch_timestamp = _utc_now_iso()

    if destination_path.exists() and any(destination_path.rglob("*")) and not force:
        return AuxFetchArtifact(
            source_name=source_name,
            source_url=f"https://huggingface.co/datasets/{repo_id}",
            repo_id=repo_id,
            version=info.sha,
            local_path=str(destination_path),
            file_size=_directory_size_bytes(destination_path),
            checksum=info.sha,
            fetch_timestamp=fetch_timestamp,
            status="reused_existing",
            usability=usability,
            note=note,
        )

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=info.sha,
        local_dir=destination_path,
        local_dir_use_symlinks=False,
        allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
        force_download=force,
    )
    return AuxFetchArtifact(
        source_name=source_name,
        source_url=f"https://huggingface.co/datasets/{repo_id}",
        repo_id=repo_id,
        version=info.sha,
        local_path=str(destination_path),
        file_size=_directory_size_bytes(destination_path),
        checksum=info.sha,
        fetch_timestamp=fetch_timestamp,
        status="snapshot_downloaded",
        usability=usability,
        note=note,
    )


def record_local_aux_source(
    *,
    source_name: str,
    path: str | Path,
    source_url: str | None,
    version: str = "local",
    usability: str = "usable",
    note: str | None = None,
) -> AuxFetchArtifact:
    """Record an already-local auxiliary source in the fetch lock."""
    path_obj = Path(path)
    checksum: str | None
    if path_obj.exists() and path_obj.is_file():
        checksum = compute_file_sha256(path_obj)
        local_path = str(path_obj)
        file_size = path_obj.stat().st_size
        status = "reused_local"
    elif path_obj.exists():
        checksum = None
        local_path = str(path_obj)
        file_size = _directory_size_bytes(path_obj)
        status = "reused_local"
    else:
        checksum = None
        local_path = None
        file_size = None
        status = "unavailable_local"
        usability = "audited_only"
    return AuxFetchArtifact(
        source_name=source_name,
        source_url=source_url,
        repo_id=None,
        version=version,
        local_path=local_path,
        file_size=file_size,
        checksum=checksum,
        fetch_timestamp=_utc_now_iso(),
        status=status,
        usability=usability,
        note=note,
    )


def record_unavailable_aux_source(
    *,
    source_name: str,
    source_url: str | None,
    repo_id: str | None,
    note: str,
) -> AuxFetchArtifact:
    """Record an unavailable or audited-only auxiliary source."""
    return AuxFetchArtifact(
        source_name=source_name,
        source_url=source_url,
        repo_id=repo_id,
        version="unknown",
        local_path=None,
        file_size=None,
        checksum=None,
        fetch_timestamp=_utc_now_iso(),
        status="unavailable",
        usability="audited_only",
        note=note,
    )


def write_aux_fetch_lock(
    *,
    entries: list[AuxFetchArtifact],
    manifest_path: str | Path,
    notes_path: str | Path,
) -> dict[str, Any]:
    """Write auxiliary-source fetch lock manifest and notes."""
    payload = {
        "generated_at": _utc_now_iso(),
        "artifacts": [asdict(entry) for entry in entries],
    }
    write_yaml(manifest_path, payload)

    notes_lines = ["# Auxiliary Fetch Notes", ""]
    for entry in entries:
        detail = entry.note or "no additional notes"
        notes_lines.append(
            f"- `{entry.source_name}`: status={entry.status}, usability={entry.usability}, note={detail}"
        )
    notes_output = Path(notes_path)
    notes_output.parent.mkdir(parents=True, exist_ok=True)
    notes_output.write_text("\n".join(notes_lines) + "\n", encoding="utf-8")
    return payload
