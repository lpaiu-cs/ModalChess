from torch.utils.data import DataLoader

from modalchess.data.collators import collate_position_samples
from modalchess.data.dataset_builder import DatasetBuildConfig, build_fixture_dataset
from modalchess.models.modalchess_core import ModalChessCoreModel
from modalchess.train.optim import build_optimizer
from modalchess.train.trainer import Trainer


def test_tiny_overfit_smoke_run() -> None:
    dataset = build_fixture_dataset(DatasetBuildConfig(history_length=1, limit=1))
    concept_vocab = [
        "check",
        "capture",
        "recapture",
        "pin",
        "fork",
        "skewer",
        "discovered_attack",
        "discovered_check",
        "king_safety",
        "passed_pawn",
        "open_file",
        "promotion_threat",
    ]
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda samples: collate_position_samples(samples, concept_vocab=concept_vocab),
    )
    model = ModalChessCoreModel(
        history_length=1,
        input_channels=18,
        d_model=64,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.0,
        use_relation_bias=True,
        concept_vocab=concept_vocab,
    )
    optimizer = build_optimizer(model, learning_rate=0.01, weight_decay=0.0)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_weights={
            "policy": 1.0,
            "state_probe": 1.0,
            "legality": 0.25,
            "value": 0.1,
            "concept": 0.1,
        },
        grad_clip_norm=1.0,
    )
    metrics = trainer.overfit(dataloader, num_steps=12)
    assert metrics["final_loss"] < metrics["initial_loss"]
