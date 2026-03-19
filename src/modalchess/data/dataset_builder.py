"""fixture 기반 학습/평가를 위한 데이터셋 빌더."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import Dataset

from modalchess.data.board_state import board_to_board_state
from modalchess.data.fixtures import DEFAULT_CONCEPT_VOCAB, fixture_boards
from modalchess.data.schema import PositionSample
from modalchess.data.tensor_codec import encode_fen_history


@dataclass(slots=True)
class DatasetBuildConfig:
    """로컬 fixture 데이터셋 생성을 위한 단순 설정."""

    history_length: int = 1
    limit: int | None = None


class FixtureDataset(Dataset[PositionSample]):
    """엄선된 fixture 포지션으로 구성한 소형 저장소 내 데이터셋."""

    def __init__(self, samples: list[PositionSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PositionSample:
        return self.samples[index]


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
        sample = PositionSample(
            position_id=spec.position_id,
            fen=board.fen(en_passant="fen"),
            history_fens=history,
            board_planes=encode_fen_history(history, history_length=config.history_length),
            meta={
                "side_to_move": "w" if board.turn else "b",
                "halfmove_clock": board.halfmove_clock,
                "fullmove_number": board.fullmove_number,
            },
            legal_moves_uci=legal_moves_uci,
            target_move_uci=spec.target_move_uci,
            next_fen=next_fen,
            concept_tags=spec.concept_tags,
            engine_eval_cp=spec.engine_eval_cp,
            board_state=board_to_board_state(board),
        )
        samples.append(sample)
    if config.limit is not None:
        return samples[: config.limit]
    return samples


def build_fixture_dataset(config: DatasetBuildConfig) -> FixtureDataset:
    """로컬 fixture 샘플을 torch Dataset으로 감싼다."""
    return FixtureDataset(build_fixture_samples(config))


def default_concept_vocab() -> list[str]:
    """기본 concept 어휘 목록을 반환한다."""
    return list(DEFAULT_CONCEPT_VOCAB)
