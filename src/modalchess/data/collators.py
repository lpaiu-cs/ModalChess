"""fixture 기반 ModalChess 학습용 배치 collate 유틸리티."""

from __future__ import annotations

from typing import Any

import chess
import torch

from modalchess.data.board_state import board_state_to_board
from modalchess.data.move_codec import uci_to_factorized
from modalchess.data.schema import PositionSample
from modalchess.data.tensor_codec import build_legality_matrix, build_state_probe_targets


def _concept_targets(concept_tags: list[str], concept_vocab: list[str]) -> torch.Tensor:
    vocab_index = {concept: idx for idx, concept in enumerate(concept_vocab)}
    targets = torch.zeros(len(concept_vocab), dtype=torch.float32)
    for concept in concept_tags:
        if concept in vocab_index:
            targets[vocab_index[concept]] = 1.0
    return targets


def collate_position_samples(
    samples: list[PositionSample],
    concept_vocab: list[str],
) -> dict[str, Any]:
    """fixture 샘플을 학습/평가 배치로 묶는다."""
    board_planes = torch.stack([sample.board_planes for sample in samples], dim=0)
    piece_planes: list[torch.Tensor] = []
    side_to_move: list[torch.Tensor] = []
    castling: list[torch.Tensor] = []
    en_passant_index: list[torch.Tensor] = []
    in_check: list[torch.Tensor] = []
    legality: list[torch.Tensor] = []
    src_targets: list[int] = []
    dst_targets: list[int] = []
    promo_targets: list[int] = []
    value_targets: list[float] = []
    concept_targets: list[torch.Tensor] = []
    legal_moves_factorized: list[list[tuple[int, int, int]]] = []

    for sample in samples:
        if sample.board_state is None:
            raise ValueError("PositionSample.board_state must be populated for batching")
        state_targets = build_state_probe_targets(sample.board_state)
        piece_planes.append(state_targets["piece_planes"])
        side_to_move.append(state_targets["side_to_move"])
        castling.append(state_targets["castling_rights"])
        en_passant_index.append(state_targets["en_passant_index"])
        in_check.append(state_targets["in_check"])
        board = board_state_to_board(sample.board_state)
        legality.append(build_legality_matrix(board))

        if sample.target_move_uci is None:
            src_targets.append(-100)
            dst_targets.append(-100)
            promo_targets.append(-100)
        else:
            factorized = uci_to_factorized(sample.target_move_uci)
            src_targets.append(factorized.src_square)
            dst_targets.append(factorized.dst_square)
            promo_targets.append(factorized.promotion)

        value_targets.append(float((sample.engine_eval_cp or 0.0) / 1000.0))
        concept_targets.append(_concept_targets(sample.concept_tags, concept_vocab))
        legal_moves_factorized.append(
            [
                (
                    move.from_square,
                    move.to_square,
                    uci_to_factorized(move.uci()).promotion,
                )
                for move in board.legal_moves
            ]
        )

    return {
        "position_ids": [sample.position_id for sample in samples],
        "fens": [sample.fen for sample in samples],
        "board_planes": board_planes,
        "legal_moves_uci": [sample.legal_moves_uci for sample in samples],
        "legal_moves_factorized": legal_moves_factorized,
        "target_move_uci": [sample.target_move_uci for sample in samples],
        "src_targets": torch.tensor(src_targets, dtype=torch.long),
        "dst_targets": torch.tensor(dst_targets, dtype=torch.long),
        "promo_targets": torch.tensor(promo_targets, dtype=torch.long),
        "state_piece_planes": torch.stack(piece_planes, dim=0),
        "state_side_to_move": torch.stack(side_to_move, dim=0),
        "state_castling_rights": torch.stack(castling, dim=0),
        "state_en_passant": torch.stack(en_passant_index, dim=0),
        "state_in_check": torch.stack(in_check, dim=0),
        "legality_matrix": torch.stack(legality, dim=0),
        "value_targets": torch.tensor(value_targets, dtype=torch.float32),
        "concept_targets": torch.stack(concept_targets, dim=0),
    }
