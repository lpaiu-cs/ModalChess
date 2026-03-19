import json
from pathlib import Path

import torch

from modalchess.data.collators import collate_position_samples
from modalchess.data.dataset_builder import DatasetBuildConfig, build_dataset
from modalchess.models.modalchess_core import ModalChessCoreModel
from modalchess.train.losses import compute_modalchess_losses


def test_missing_auxiliary_labels_are_masked_from_losses(tmp_path: Path) -> None:
    dataset_path = tmp_path / "missing_aux.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "position_id": "p1",
                    "game_id": "g1",
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "target_move_uci": "e2e4",
                }
            )
            + "\n"
        )

    dataset = build_dataset(
        DatasetBuildConfig(
            source="jsonl",
            dataset_path=str(dataset_path),
            split="all",
        )
    )
    batch = collate_position_samples(dataset.samples, concept_vocab=["check"])
    assert not bool(batch["has_engine_eval"].item())
    assert not bool(batch["has_concept_labels"].item())

    model = ModalChessCoreModel(
        history_length=1,
        input_channels=18,
        d_model=32,
        num_layers=1,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.0,
        use_relation_bias=True,
        concept_vocab=["check"],
    )
    outputs = model(
        board_planes=batch["board_planes"],
        meta_features=batch["meta_features"],
    )
    losses = compute_modalchess_losses(
        outputs,
        batch,
        weights={
            "policy": 1.0,
            "policy_axis_ce": 1.0,
            "policy_listwise": 1.0,
            "state_probe": 1.0,
            "legality": 0.25,
            "value": 0.1,
            "concept": 0.1,
        },
    )
    assert torch.isclose(losses["value_loss"], torch.tensor(0.0))
    assert torch.isclose(losses["concept_loss"], torch.tensor(0.0))
