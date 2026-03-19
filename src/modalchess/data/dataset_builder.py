"""fixture 기반 학습/평가를 위한 데이터셋 빌더."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path

import chess
from torch.utils.data import Dataset

from modalchess.data.board_state import board_state_to_board, board_to_board_state
from modalchess.data.fen_codec import fen_to_board_state
from modalchess.data.fixtures import DEFAULT_CONCEPT_VOCAB, fixture_boards
from modalchess.data.preprocessing_common import assert_history_fens_contract
from modalchess.data.schema import PositionSample
from modalchess.data.tensor_codec import encode_fen_history


@dataclass(slots=True)
class DatasetBuildConfig:
    """로컬 fixture 데이터셋 생성을 위한 단순 설정."""

    source: str = "fixture"
    history_length: int = 1
    limit: int | None = None
    dataset_path: str | None = None
    split: str = "all"
    split_seed: int = 7
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    split_field: str | None = None
    require_repetition_count: bool = False
    allow_position_level_split: bool = False


class FixtureDataset(Dataset[PositionSample]):
    """엄선된 fixture 포지션으로 구성한 소형 저장소 내 데이터셋."""

    def __init__(self, samples: list[PositionSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PositionSample:
        return self.samples[index]


def validate_position_sample(
    sample: PositionSample,
    require_repetition_count: bool = False,
    repetition_count_present: bool = True,
) -> None:
    """샘플이 학습 전 만족해야 하는 일관성 규칙을 검증한다."""
    assert_history_fens_contract(sample.position_id, sample.fen, sample.history_fens)
    board = board_state_to_board(sample.board_state)
    if sample.next_fen is not None and sample.target_move_uci is None:
        raise ValueError(
            f"next_fen을 제공하는 레코드는 target_move_uci도 함께 가져야 한다: {sample.position_id}"
        )
    if sample.target_move_uci is not None:
        legal_moves = {move.uci() for move in board.legal_moves}
        if sample.target_move_uci not in legal_moves:
            raise ValueError(
                f"target_move_uci가 합법 수가 아니다: {sample.position_id} / {sample.target_move_uci}"
            )
        if sample.next_fen is not None:
            next_board = board.copy(stack=False)
            next_board.push_uci(sample.target_move_uci)
            expected_next_fen = next_board.fen(en_passant="fen")
            if sample.next_fen != expected_next_fen:
                raise ValueError(
                    f"next_fen이 target_move_uci 결과와 일치하지 않는다: {sample.position_id}"
                )
    if require_repetition_count and not repetition_count_present:
        raise ValueError(f"repetition_count가 필요한 실험인데 누락됐다: {sample.position_id}")


def build_fixture_samples(config: DatasetBuildConfig) -> list[PositionSample]:
    """스모크 테스트와 베이스라인 루프용 로컬 fixture 샘플을 만든다."""
    samples: list[PositionSample] = []
    for spec, board, history in fixture_boards():
        legal_moves_uci = [move.uci() for move in board.legal_moves]
        next_fen = None
        if spec.target_move_uci is not None:
            next_board = board.copy(stack=False)
            next_board.push_uci(spec.target_move_uci)
            next_fen = next_board.fen(en_passant="fen")
        board_state = board_to_board_state(board)
        sample = PositionSample(
            position_id=spec.position_id,
            game_id=spec.position_id,
            fen=board.fen(en_passant="fen"),
            history_fens=history,
            board_planes=encode_fen_history(history, history_length=config.history_length),
            legal_moves_uci=legal_moves_uci,
            board_state=board_state,
            target_move_uci=spec.target_move_uci,
            next_fen=next_fen,
            concept_tags=spec.concept_tags,
            engine_eval_cp=spec.engine_eval_cp,
        )
        validate_position_sample(
            sample,
            require_repetition_count=config.require_repetition_count,
            repetition_count_present=False,
        )
        samples.append(sample)
    if config.limit is not None:
        return samples[: config.limit]
    return samples


def build_fixture_dataset(config: DatasetBuildConfig) -> FixtureDataset:
    """로컬 fixture 샘플을 torch Dataset으로 감싼다."""
    return FixtureDataset(build_fixture_samples(config))


def _split_by_game_id(
    samples: list[PositionSample],
    config: DatasetBuildConfig,
) -> list[PositionSample]:
    if config.split != "all" and not config.allow_position_level_split:
        missing_game_ids = [sample.position_id for sample in samples if sample.game_id is None]
        if missing_game_ids:
            raise ValueError(
                "game-level split에는 모든 샘플의 game_id가 필요하다. "
                "position 단위 분할이 필요하면 allow_position_level_split=true를 명시해야 한다."
            )
    game_ids = sorted(
        {
            sample.game_id if sample.game_id is not None else sample.position_id
            for sample in samples
        }
    )
    rng = random.Random(config.split_seed)
    rng.shuffle(game_ids)
    train_cut = int(len(game_ids) * config.train_ratio)
    val_cut = train_cut + int(len(game_ids) * config.val_ratio)
    split_to_ids = {
        "train": set(game_ids[:train_cut]),
        "val": set(game_ids[train_cut:val_cut]),
        "test": set(game_ids[val_cut:]),
        "all": set(game_ids),
    }
    selected_ids = split_to_ids.get(config.split)
    if selected_ids is None:
        raise ValueError(f"지원하지 않는 split: {config.split}")
    filtered = [
        sample
        for sample in samples
        if (
            sample.game_id if sample.game_id is not None else sample.position_id
        ) in selected_ids
    ]
    return filtered


def _load_jsonl_records(dataset_path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _resolve_split_field(records: list[dict[str, object]], config: DatasetBuildConfig) -> str | None:
    if config.split_field is not None:
        return config.split_field
    if not records:
        return None
    split_present = [("split" in record) for record in records]
    if any(split_present) and not all(split_present):
        raise ValueError("split 필드는 모든 JSONL 레코드에 일관되게 존재해야 한다.")
    if all(split_present):
        return "split"
    return None


def _filter_records_by_split_field(
    records: list[dict[str, object]],
    split_field: str,
    split_name: str,
) -> list[dict[str, object]]:
    supported_splits = {"train", "val", "test", "all"}
    if split_name not in supported_splits:
        raise ValueError(f"지원하지 않는 split: {split_name}")
    if split_name == "all":
        return records
    filtered: list[dict[str, object]] = []
    for record in records:
        if split_field not in record:
            raise ValueError(f"{split_field} 필드가 누락된 JSONL 레코드가 있다: {record['position_id']}")
        record_split = record[split_field]
        if record_split not in {"train", "val", "test"}:
            raise ValueError(
                f"{split_field} 값은 train/val/test 중 하나여야 한다: "
                f"{record['position_id']} / {record_split}"
            )
        if record_split == split_name:
            filtered.append(record)
    return filtered


def build_jsonl_samples(config: DatasetBuildConfig) -> list[PositionSample]:
    """JSONL 기반 실제 연구 데이터 샘플을 생성한다."""
    if config.dataset_path is None:
        raise ValueError("JSONL 데이터셋에는 dataset_path가 필요하다.")
    dataset_path = Path(config.dataset_path)
    records = _load_jsonl_records(dataset_path)
    split_field = _resolve_split_field(records, config)
    if split_field is not None:
        records = _filter_records_by_split_field(records, split_field, config.split)
    samples: list[PositionSample] = []
    for record in records:
        fen = record["fen"]
        history_fens = record.get("history_fens") or [fen]
        board_state = fen_to_board_state(fen)
        board = board_state_to_board(board_state)
        computed_legal_moves_uci = [move.uci() for move in board.legal_moves]
        provided_legal_moves_uci = record.get("legal_moves_uci")
        if provided_legal_moves_uci is not None:
            if (
                len(provided_legal_moves_uci) != len(computed_legal_moves_uci)
                or set(provided_legal_moves_uci) != set(computed_legal_moves_uci)
            ):
                raise ValueError(
                    "legal_moves_uci가 python-chess 합법 수 집합과 일치하지 않는다: "
                    f"{record['position_id']}"
                )
        sample = PositionSample(
            position_id=record["position_id"],
            game_id=record.get("game_id"),
            fen=fen,
            history_fens=history_fens,
            board_planes=encode_fen_history(history_fens, history_length=config.history_length),
            legal_moves_uci=computed_legal_moves_uci,
            board_state=board_state,
            target_move_uci=record.get("target_move_uci"),
            next_fen=record.get("next_fen"),
            concept_tags=record["concept_tags"] if "concept_tags" in record else None,
            engine_eval_cp=record.get("engine_eval_cp"),
        )
        repetition_count_present = "repetition_count" in record
        if repetition_count_present:
            sample.board_state.meta.repetition_count = int(record["repetition_count"])
        validate_position_sample(
            sample,
            require_repetition_count=config.require_repetition_count,
            repetition_count_present=repetition_count_present,
        )
        samples.append(sample)
    if split_field is None:
        samples = _split_by_game_id(samples, config)
    if config.limit is not None:
        return samples[: config.limit]
    return samples


def build_jsonl_dataset(config: DatasetBuildConfig) -> FixtureDataset:
    """JSONL 샘플을 torch Dataset으로 감싼다."""
    return FixtureDataset(build_jsonl_samples(config))


def build_dataset(config: DatasetBuildConfig) -> FixtureDataset:
    """데이터 소스에 따라 학습/평가용 데이터셋을 생성한다."""
    if config.source == "fixture":
        return build_fixture_dataset(config)
    if config.source == "jsonl":
        return build_jsonl_dataset(config)
    raise ValueError(f"지원하지 않는 데이터 소스: {config.source}")


def default_concept_vocab() -> list[str]:
    """기본 concept 어휘 목록을 반환한다."""
    return list(DEFAULT_CONCEPT_VOCAB)
