"""fixture 기반 ModalChess 학습용 배치 collate 유틸리티."""

from __future__ import annotations

from typing import Any

import chess
import torch

from modalchess.data.board_state import board_state_to_board
from modalchess.data.move_codec import move_to_factorized, uci_to_factorized
from modalchess.data.schema import PositionSample
from modalchess.data.tensor_codec import build_legality_tensor, build_state_probe_targets


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
    square_state: list[torch.Tensor] = []
    side_to_move: list[torch.Tensor] = []
    castling: list[torch.Tensor] = []
    en_passant_index: list[torch.Tensor] = []
    in_check: list[torch.Tensor] = []
    legality: list[torch.Tensor] = []
    meta_features: list[torch.Tensor] = []
    src_targets: list[int] = []
    dst_targets: list[int] = []
    promo_targets: list[int] = []
    target_legal_move_index: list[int] = []
    value_targets: list[float] = []
    concept_targets: list[torch.Tensor] = []
    legal_moves_factorized: list[list[tuple[int, int, int]]] = []
    subset_promotion: list[bool] = []
    subset_castling: list[bool] = []
    subset_en_passant: list[bool] = []
    subset_check_evasion: list[bool] = []
    target_is_promotion: list[bool] = []
    target_is_castling: list[bool] = []
    target_is_en_passant: list[bool] = []

    for sample in samples:
        state_targets = build_state_probe_targets(sample.board_state)
        square_state.append(state_targets["square_state"])
        side_to_move.append(state_targets["side_to_move"])
        castling.append(state_targets["castling_rights"])
        en_passant_index.append(state_targets["en_passant_index"])
        in_check.append(state_targets["in_check"])
        board = board_state_to_board(sample.board_state)
        legality.append(build_legality_tensor(board))
        meta_features.append(
            torch.tensor(
                [
                    float(sample.meta.halfmove_clock),
                    float(sample.meta.fullmove_number),
                    float(sample.meta.repetition_count),
                ],
                dtype=torch.float32,
            )
        )
        legal_moves = [move_to_factorized(move) for move in board.legal_moves]
        legal_move_tuples = [
            (move.src_square, move.dst_square, move.promotion)
            for move in legal_moves
        ]

        if sample.target_move_uci is None:
            src_targets.append(-100)
            dst_targets.append(-100)
            promo_targets.append(-100)
            target_legal_move_index.append(-100)
            target_is_promotion.append(False)
            target_is_castling.append(False)
            target_is_en_passant.append(False)
        else:
            factorized = uci_to_factorized(sample.target_move_uci)
            target_move = chess.Move.from_uci(sample.target_move_uci)
            src_targets.append(factorized.src_square)
            dst_targets.append(factorized.dst_square)
            promo_targets.append(factorized.promotion)
            target_legal_move_index.append(
                legal_move_tuples.index(
                    (factorized.src_square, factorized.dst_square, factorized.promotion)
                )
            )
            target_is_promotion.append(factorized.promotion != 0)
            target_is_castling.append(board.is_castling(target_move))
            target_is_en_passant.append(board.is_en_passant(target_move))

        value_targets.append(float((sample.engine_eval_cp or 0.0) / 1000.0))
        concept_targets.append(_concept_targets(sample.concept_tags, concept_vocab))
        legal_moves_factorized.append(legal_move_tuples)
        subset_promotion.append(any(move.promotion != 0 for move in legal_moves))
        subset_castling.append(any(board.is_castling(move_to_factorized_move) for move_to_factorized_move in board.legal_moves))
        subset_en_passant.append(any(board.is_en_passant(move) for move in board.legal_moves))
        subset_check_evasion.append(board.is_check())

    return {
        "position_ids": [sample.position_id for sample in samples],
        "fens": [sample.fen for sample in samples],
        "board_planes": board_planes,
        "meta_features": torch.stack(meta_features, dim=0),
        "legal_moves_uci": [sample.legal_moves_uci for sample in samples],
        "legal_moves_factorized": legal_moves_factorized,
        "target_move_uci": [sample.target_move_uci for sample in samples],
        "src_targets": torch.tensor(src_targets, dtype=torch.long),
        "dst_targets": torch.tensor(dst_targets, dtype=torch.long),
        "promo_targets": torch.tensor(promo_targets, dtype=torch.long),
        "target_legal_move_index": torch.tensor(target_legal_move_index, dtype=torch.long),
        "state_square_classes": torch.stack(square_state, dim=0),
        "state_side_to_move": torch.stack(side_to_move, dim=0),
        "state_castling_rights": torch.stack(castling, dim=0),
        "state_en_passant": torch.stack(en_passant_index, dim=0),
        "state_in_check": torch.stack(in_check, dim=0),
        "legality_tensor": torch.stack(legality, dim=0),
        "value_targets": torch.tensor(value_targets, dtype=torch.float32),
        "concept_targets": torch.stack(concept_targets, dim=0),
        "subset_promotion": torch.tensor(subset_promotion, dtype=torch.bool),
        "subset_castling": torch.tensor(subset_castling, dtype=torch.bool),
        "subset_en_passant": torch.tensor(subset_en_passant, dtype=torch.bool),
        "subset_check_evasion": torch.tensor(subset_check_evasion, dtype=torch.bool),
        "target_is_promotion": torch.tensor(target_is_promotion, dtype=torch.bool),
        "target_is_castling": torch.tensor(target_is_castling, dtype=torch.bool),
        "target_is_en_passant": torch.tensor(target_is_en_passant, dtype=torch.bool),
    }
