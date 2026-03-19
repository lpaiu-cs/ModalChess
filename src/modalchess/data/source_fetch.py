"""실제 파일럿용 raw source fetch/snapshot 유틸리티."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download
import requests

from modalchess.data.preprocessing_common import compute_file_sha256, write_yaml


LICHESS_ROOT = "https://database.lichess.org"
LICHESS_STANDARD_LIST_URL = f"{LICHESS_ROOT}/standard/list.txt"
LICHESS_STANDARD_SHA256_URL = f"{LICHESS_ROOT}/standard/sha256sums.txt"
LICHESS_PUZZLE_URL = f"{LICHESS_ROOT}/lichess_db_puzzle.csv.zst"
LICHESS_EVAL_URL = f"{LICHESS_ROOT}/lichess_db_eval.jsonl.zst"


@dataclass(slots=True)
class FetchedArtifact:
    """fetch lock manifest에 저장할 단일 artifact 메타데이터."""

    source_name: str
    source_url: str
    version: str
    local_path: str | None
    file_size: int | None
    checksum: str | None
    fetch_timestamp: str
    status: str
    note: str | None = None


def utc_now_iso() -> str:
    """UTC 기준 현재 시각을 ISO 문자열로 반환한다."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _http_head(url: str) -> dict[str, str]:
    response = requests.head(url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    return dict(response.headers)


def inspect_url_metadata(url: str) -> dict[str, str]:
    """공개 HTTP artifact의 HEAD 메타데이터를 반환한다."""
    return _http_head(url)


def _parse_lichess_sha256_map(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        checksum, filename = stripped.split(maxsplit=1)
        mapping[filename.strip()] = checksum.strip()
    return mapping


def fetch_lichess_standard_versions() -> list[str]:
    """Lichess standard list에서 사용 가능한 월 버전을 반환한다."""
    response = requests.get(LICHESS_STANDARD_LIST_URL, timeout=60)
    response.raise_for_status()
    versions: list[str] = []
    for line in response.text.splitlines():
        filename = line.rstrip("/").split("/")[-1]
        if filename.startswith("lichess_db_standard_rated_") and filename.endswith(".pgn.zst"):
            versions.append(filename.removeprefix("lichess_db_standard_rated_").removesuffix(".pgn.zst"))
    return versions


def resolve_lichess_standard_url(version: str) -> tuple[str, str]:
    """standard PGN 버전을 URL과 정규화된 version 문자열로 바꾼다."""
    if version == "latest":
        versions = fetch_lichess_standard_versions()
        if not versions:
            raise RuntimeError("Lichess standard version 목록을 가져오지 못했다.")
        resolved = versions[0]
    else:
        resolved = version
    filename = f"lichess_db_standard_rated_{resolved}.pgn.zst"
    return f"{LICHESS_ROOT}/standard/{filename}", resolved


def lookup_lichess_standard_checksum(version: str) -> str | None:
    """standard sha256sums.txt에서 특정 월의 checksum을 조회한다."""
    response = requests.get(LICHESS_STANDARD_SHA256_URL, timeout=60)
    response.raise_for_status()
    checksum_map = _parse_lichess_sha256_map(response.text)
    return checksum_map.get(f"lichess_db_standard_rated_{version}.pgn.zst")


def download_url_artifact(
    *,
    source_name: str,
    url: str,
    version: str,
    destination: str | Path,
    force: bool = False,
    checksum_hint: str | None = None,
) -> FetchedArtifact:
    """HTTP artifact를 로컬로 내려받고 메타데이터를 반환한다."""
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    fetch_timestamp = utc_now_iso()
    headers = _http_head(url)

    if destination_path.exists() and not force:
        return FetchedArtifact(
            source_name=source_name,
            source_url=url,
            version=version,
            local_path=str(destination_path),
            file_size=destination_path.stat().st_size,
            checksum=checksum_hint or compute_file_sha256(destination_path),
            fetch_timestamp=fetch_timestamp,
            status="reused_existing",
        )

    temp_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    os.replace(temp_path, destination_path)
    return FetchedArtifact(
        source_name=source_name,
        source_url=url,
        version=version,
        local_path=str(destination_path),
        file_size=destination_path.stat().st_size,
        checksum=checksum_hint or compute_file_sha256(destination_path),
        fetch_timestamp=fetch_timestamp,
        status="downloaded",
        note=headers.get("Last-Modified"),
    )


def snapshot_hf_dataset(
    *,
    source_name: str,
    repo_id: str,
    destination: str | Path,
    force: bool = False,
    allow_patterns: Iterable[str] | None = None,
) -> FetchedArtifact:
    """HF dataset repo를 pinned sha로 snapshot 다운로드한다."""
    api = HfApi()
    info = api.dataset_info(repo_id)
    destination_path = Path(destination)
    fetch_timestamp = utc_now_iso()

    if destination_path.exists() and any(destination_path.rglob("*")) and not force:
        return FetchedArtifact(
            source_name=source_name,
            source_url=f"https://huggingface.co/datasets/{repo_id}",
            version=info.sha,
            local_path=str(destination_path),
            file_size=None,
            checksum=info.sha,
            fetch_timestamp=fetch_timestamp,
            status="reused_existing",
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
    return FetchedArtifact(
        source_name=source_name,
        source_url=f"https://huggingface.co/datasets/{repo_id}",
        version=info.sha,
        local_path=str(destination_path),
        file_size=None,
        checksum=info.sha,
        fetch_timestamp=fetch_timestamp,
        status="snapshot_downloaded",
    )


def record_snapshot_only_entry(
    *,
    source_name: str,
    url: str,
    version: str,
    note: str,
) -> FetchedArtifact:
    """너무 큰 원천 등으로 실제 materialize 대신 lock만 남기는 entry."""
    headers = _http_head(url)
    return FetchedArtifact(
        source_name=source_name,
        source_url=url,
        version=version,
        local_path=None,
        file_size=int(headers["Content-Length"]) if headers.get("Content-Length") else None,
        checksum=None,
        fetch_timestamp=utc_now_iso(),
        status="snapshot_only",
        note=note,
    )


def record_manual_entry(
    *,
    source_name: str,
    url: str,
    version: str,
    note: str,
) -> FetchedArtifact:
    """스크립트 fetch가 불안정할 때 manual note용 entry."""
    return FetchedArtifact(
        source_name=source_name,
        source_url=url,
        version=version,
        local_path=None,
        file_size=None,
        checksum=None,
        fetch_timestamp=utc_now_iso(),
        status="manual_required",
        note=note,
    )


def write_fetch_lock_manifest(
    *,
    entries: list[FetchedArtifact],
    manifest_path: str | Path,
    notes_path: str | Path,
) -> dict[str, Any]:
    """raw fetch lock manifest와 human-readable note를 기록한다."""
    payload = {
        "generated_at": utc_now_iso(),
        "artifacts": [asdict(entry) for entry in entries],
    }
    write_yaml(manifest_path, payload)

    notes_lines = ["# Raw Fetch Notes", ""]
    note_entries = [entry for entry in entries if entry.note]
    if not note_entries:
        notes_lines.append("All configured sources were materialized without extra notes.")
    else:
        for entry in note_entries:
            notes_lines.append(f"- `{entry.source_name}`: {entry.note}")
    notes_output = Path(notes_path)
    notes_output.parent.mkdir(parents=True, exist_ok=True)
    notes_output.write_text("\n".join(notes_lines) + "\n", encoding="utf-8")
    return payload
