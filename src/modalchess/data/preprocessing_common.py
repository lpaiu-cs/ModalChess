"""Raw-data 전처리 파이프라인에서 공유하는 유틸리티."""

from __future__ import annotations

import bz2
import csv
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import lzma
from pathlib import Path
import pickle
from typing import Any, Iterable, Iterator, Mapping, TextIO
import zipfile

import chess
import yaml


COMPRESSION_SUFFIXES = {".gz", ".bz2", ".xz", ".zst"}


@dataclass(slots=True)
class StableSplitConfig:
    """game_id 해시 기반 split 설정."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    salt: str = "modalchess"

    def validate(self) -> None:
        if self.train_ratio <= 0.0 or self.val_ratio < 0.0:
            raise ValueError("train_ratio는 양수, val_ratio는 음수가 아니어야 한다.")
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio는 1.0보다 작아야 한다.")


def stable_hash_text(
    text: str,
    prefix: str = "",
    length: int = 16,
) -> str:
    """문자열로부터 짧고 안정적인 해시 식별자를 만든다."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}{digest}"


def stable_hash_record(
    record: Mapping[str, Any],
    prefix: str = "",
    length: int = 16,
) -> str:
    """정렬된 JSON 표현을 기반으로 레코드 해시를 만든다."""
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)
    return stable_hash_text(canonical, prefix=prefix, length=length)


def assign_split_by_game_id(
    game_id: str,
    split_config: StableSplitConfig | None = None,
) -> str:
    """game_id 해시로 train/val/test split을 안정적으로 배정한다."""
    config = split_config or StableSplitConfig()
    config.validate()
    digest = hashlib.sha256(f"{config.salt}:{game_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big") / float(1 << 64)
    if bucket < config.train_ratio:
        return "train"
    if bucket < config.train_ratio + config.val_ratio:
        return "val"
    return "test"


def normalize_fen_for_eval_join(fen: str) -> str:
    """평가 조인을 위해 FEN의 첫 4개 필드만 남긴다."""
    parts = fen.strip().split()
    if len(parts) < 4:
        raise ValueError(f"FEN이 4개 필드 미만이다: {fen}")
    return " ".join(parts[:4])


def assert_history_fens_contract(position_id: str, fen: str, history_fens: list[str]) -> None:
    """history_fens가 ModalChess 계약을 만족하는지 검증한다."""
    if not history_fens:
        raise ValueError(f"history_fens가 비어 있다: {position_id}")
    if history_fens[-1] != fen:
        raise ValueError(f"history_fens[-1]이 현재 fen과 일치하지 않는다: {position_id}")
    for transition_index, (previous_fen, current_fen) in enumerate(
        zip(history_fens, history_fens[1:], strict=False)
    ):
        board = chess.Board(previous_fen)
        reachable = False
        for move in board.legal_moves:
            next_board = board.copy(stack=False)
            next_board.push(move)
            if next_board.fen(en_passant="fen") == current_fen:
                reachable = True
                break
        if not reachable:
            raise ValueError(
                "history_fens 전이가 합법적인 단일 수로 연결되지 않는다: "
                f"{position_id} / step={transition_index}"
            )


def validate_modalchess_record(
    record: Mapping[str, Any],
    require_target_move: bool = False,
) -> None:
    """원시 파이프라인이 생성한 단일 ModalChess 레코드를 검증한다."""
    required_fields = ("position_id", "game_id", "fen")
    missing_fields = [field_name for field_name in required_fields if not record.get(field_name)]
    if missing_fields:
        raise ValueError(f"필수 필드 누락: {missing_fields}")
    fen = str(record["fen"])
    board = chess.Board(fen)

    history_fens_raw = record.get("history_fens")
    if history_fens_raw is not None:
        if not isinstance(history_fens_raw, list):
            raise ValueError(f"history_fens는 list여야 한다: {record['position_id']}")
        assert_history_fens_contract(str(record["position_id"]), fen, [str(item) for item in history_fens_raw])

    target_move_uci = record.get("target_move_uci")
    if require_target_move and not target_move_uci:
        raise ValueError(f"target_move_uci가 필요한 supervised 레코드인데 누락됐다: {record['position_id']}")
    if target_move_uci is not None:
        move = chess.Move.from_uci(str(target_move_uci))
        if move not in board.legal_moves:
            raise ValueError(
                f"target_move_uci가 합법 수가 아니다: {record['position_id']} / {target_move_uci}"
            )
    if record.get("next_fen") is not None:
        if target_move_uci is None:
            raise ValueError(
                f"next_fen을 제공하는 레코드는 target_move_uci도 함께 가져야 한다: {record['position_id']}"
            )
        next_board = board.copy(stack=False)
        next_board.push_uci(str(target_move_uci))
        expected_next_fen = next_board.fen(en_passant="fen")
        if str(record["next_fen"]) != expected_next_fen:
            raise ValueError(
                f"next_fen이 target_move_uci 결과와 일치하지 않는다: {record['position_id']}"
            )
    split_name = record.get("split")
    if split_name is not None and split_name not in {"train", "val", "test"}:
        raise ValueError(f"지원하지 않는 split 값이다: {record['position_id']} / {split_name}")
    legal_moves_uci = record.get("legal_moves_uci")
    if legal_moves_uci is not None:
        legal_move_set = {move.uci() for move in board.legal_moves}
        if set(str(move) for move in legal_moves_uci) != legal_move_set:
            raise ValueError(f"legal_moves_uci가 합법 수 집합과 일치하지 않는다: {record['position_id']}")


def special_rule_flags(fen: str, target_move_uci: str | None) -> dict[str, bool]:
    """현재 포지션과 타깃 수의 special-rule 속성을 계산한다."""
    board = chess.Board(fen)
    if target_move_uci is None:
        return {
            "promotion": False,
            "castling": False,
            "en_passant": False,
            "check_evasion": board.is_check(),
        }
    move = chess.Move.from_uci(target_move_uci)
    return {
        "promotion": move.promotion is not None,
        "castling": board.is_castling(move),
        "en_passant": board.is_en_passant(move),
        "check_evasion": board.is_check(),
    }


def parse_space_or_comma_separated(value: Any) -> list[str]:
    """공백 또는 쉼표로 구분된 문자열/리스트를 표준 list[str]로 바꾼다."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "," in text and " " not in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    return [item.strip() for item in text.replace(",", " ").split() if item.strip()]


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """JSONL 파일로 레코드를 기록하고 row 수를 반환한다."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_yaml(path: str | Path, payload: Mapping[str, Any]) -> None:
    """YAML 파일을 기록한다."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False, allow_unicode=True)


def _open_text_stream_from_bytes(data: bytes, inner_name: str) -> TextIO:
    if inner_name.endswith(".gz"):
        return io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(data)), encoding="utf-8")
    if inner_name.endswith(".bz2"):
        return io.TextIOWrapper(io.BytesIO(bz2.decompress(data)), encoding="utf-8")
    if inner_name.endswith(".xz"):
        return io.TextIOWrapper(io.BytesIO(lzma.decompress(data)), encoding="utf-8")
    return io.StringIO(data.decode("utf-8"))


def open_text_input(path: str | Path) -> TextIO:
    """평문 또는 압축 텍스트 입력을 TextIO로 연다."""
    input_path = Path(path)
    suffixes = input_path.suffixes
    if not suffixes:
        return input_path.open("r", encoding="utf-8")
    if suffixes[-1] == ".gz":
        return io.TextIOWrapper(gzip.open(input_path, mode="rb"), encoding="utf-8")
    if suffixes[-1] == ".bz2":
        return io.TextIOWrapper(bz2.open(input_path, mode="rb"), encoding="utf-8")
    if suffixes[-1] == ".xz":
        return io.TextIOWrapper(lzma.open(input_path, mode="rb"), encoding="utf-8")
    if suffixes[-1] == ".zst":
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError("`.zst` 입력을 읽으려면 `zstandard` 패키지가 필요하다.") from exc
        handle = input_path.open("rb")
        decompressor = zstandard.ZstdDecompressor()
        reader = decompressor.stream_reader(handle)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return input_path.open("r", encoding="utf-8")


def _flatten_pickled_payload(payload: Any) -> list[dict[str, Any]]:
    if hasattr(payload, "to_dict"):
        try:
            converted = payload.to_dict(orient="records")
            if isinstance(converted, list):
                return [dict(row) for row in converted]
        except TypeError:
            pass
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        if "records" in payload and isinstance(payload["records"], list):
            return [dict(row) for row in payload["records"] if isinstance(row, Mapping)]
        rows: list[dict[str, Any]] = []
        for value in payload.values():
            if isinstance(value, list):
                rows.extend(dict(row) for row in value if isinstance(row, Mapping))
        if rows:
            return rows
    raise ValueError("pickle payload를 레코드 리스트로 변환할 수 없다.")


def _load_records_from_text_stream(stream: TextIO, suffix: str) -> list[dict[str, Any]]:
    if suffix == ".jsonl":
        return [json.loads(line) for line in stream if line.strip()]
    if suffix == ".json":
        payload = json.load(stream)
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, Mapping)]
        if isinstance(payload, Mapping):
            if "records" in payload and isinstance(payload["records"], list):
                return [dict(row) for row in payload["records"] if isinstance(row, Mapping)]
            raise ValueError("JSON payload에서 records 리스트를 찾지 못했다.")
        raise ValueError("지원하지 않는 JSON payload 형식이다.")
    if suffix == ".csv":
        return [dict(row) for row in csv.DictReader(stream)]
    raise ValueError(f"지원하지 않는 구조화 파일 형식이다: {suffix}")


def _load_records_from_zip(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            data = archive.read(member)
            inner_suffixes = Path(member).suffixes
            base_suffix = ""
            for suffix in reversed(inner_suffixes):
                if suffix in {".jsonl", ".json", ".csv", ".pkl", ".pickle"}:
                    base_suffix = suffix
                    break
            if base_suffix in {".jsonl", ".json", ".csv"}:
                records.extend(_load_records_from_text_stream(_open_text_stream_from_bytes(data, member), base_suffix))
                continue
            if base_suffix in {".pkl", ".pickle"}:
                records.extend(_flatten_pickled_payload(pickle.loads(data)))
    return records


def load_records_from_path(path: str | Path) -> list[dict[str, Any]]:
    """CSV/JSON/JSONL/ZIP 계열 입력을 dict 레코드 리스트로 읽는다."""
    input_path = Path(path)
    if input_path.suffix == ".zip":
        return _load_records_from_zip(input_path)
    suffixes = input_path.suffixes
    base_suffix = suffixes[-2] if suffixes and suffixes[-1] in COMPRESSION_SUFFIXES and len(suffixes) >= 2 else input_path.suffix
    with open_text_input(input_path) as handle:
        return _load_records_from_text_stream(handle, base_suffix)


def summarize_subset_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """레코드 집합에 대해 special-rule subset 카운트를 요약한다."""
    counts = {
        "promotion": 0,
        "castling": 0,
        "en_passant": 0,
        "check_evasion": 0,
    }
    for record in records:
        flags = special_rule_flags(str(record["fen"]), str(record["target_move_uci"]) if record.get("target_move_uci") else None)
        for key, is_active in flags.items():
            counts[key] += int(is_active)
    return counts


def count_by_split(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """split별 레코드 수를 센다."""
    counts = {"train": 0, "val": 0, "test": 0}
    for record in records:
        split_name = str(record.get("split", ""))
        if split_name in counts:
            counts[split_name] += 1
    return counts


def iter_records(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    """여러 입력 경로를 순회하며 레코드를 방출한다."""
    for path in paths:
        yield from load_records_from_path(path)
