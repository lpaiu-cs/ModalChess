"""scale_v1 파이프라인(lazy dataset, LR schedule, early stop, rating filter) 검증."""

from __future__ import annotations

from functools import partial
import io
import json
import pickle
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from modalchess.data.collators import collate_position_samples
from modalchess.data.dataset_builder import (
    DatasetBuildConfig,
    LazyJsonlDataset,
    build_dataset,
    build_jsonl_samples,
)
from modalchess.data.pgn_pilot import PgnPilotBuildConfig, build_supervised_records_from_pgn
from modalchess.train.optim import build_optimizer, build_lr_scheduler, warmup_cosine_lr_lambda
import modalchess.train.train_spatial_baseline as train_entry


FIXTURE_PILOT_PATH = "data/pilot/week1_fixture_pilot.jsonl"


def _load_fixture_records() -> list[dict[str, object]]:
    records = []
    with open(FIXTURE_PILOT_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_lazy_matches_eager_on_explicit_split() -> None:
    for split in ("train", "val", "all"):
        eager_config = DatasetBuildConfig(
            source="jsonl", dataset_path=FIXTURE_PILOT_PATH, split=split
        )
        lazy_config = DatasetBuildConfig(
            source="jsonl", dataset_path=FIXTURE_PILOT_PATH, split=split, loading="lazy"
        )
        eager_samples = build_jsonl_samples(eager_config)
        lazy_dataset = build_dataset(lazy_config)
        assert isinstance(lazy_dataset, LazyJsonlDataset)
        assert len(lazy_dataset) == len(eager_samples)
        for index in range(len(eager_samples)):
            eager_sample = eager_samples[index]
            lazy_sample = lazy_dataset[index]
            assert lazy_sample.position_id == eager_sample.position_id
            assert lazy_sample.fen == eager_sample.fen
            assert lazy_sample.target_move_uci == eager_sample.target_move_uci
            assert lazy_sample.legal_moves_uci == eager_sample.legal_moves_uci
            assert torch.equal(lazy_sample.board_planes, eager_sample.board_planes)


def test_lazy_matches_eager_on_ratio_split(tmp_path: Path) -> None:
    records = _load_fixture_records()
    for record in records:
        record.pop("split", None)
    dataset_path = tmp_path / "no_split.jsonl"
    _write_jsonl(dataset_path, records)
    for split in ("train", "val", "test"):
        eager = build_jsonl_samples(
            DatasetBuildConfig(source="jsonl", dataset_path=str(dataset_path), split=split)
        )
        lazy = build_dataset(
            DatasetBuildConfig(
                source="jsonl", dataset_path=str(dataset_path), split=split, loading="lazy"
            )
        )
        assert [sample.position_id for sample in eager] == [
            lazy[index].position_id for index in range(len(lazy))
        ]


def test_lazy_rejects_split_hygiene_violation(tmp_path: Path) -> None:
    records = _load_fixture_records()
    records[0]["game_id"] = "shared_game"
    records[0]["split"] = "train"
    records[1]["game_id"] = "shared_game"
    records[1]["split"] = "val"
    dataset_path = tmp_path / "leaky.jsonl"
    _write_jsonl(dataset_path, records)
    with pytest.raises(ValueError, match="여러 split"):
        build_dataset(
            DatasetBuildConfig(
                source="jsonl", dataset_path=str(dataset_path), split="train", loading="lazy"
            )
        )


def test_lazy_sampled_validation_catches_corrupt_record(tmp_path: Path) -> None:
    records = _load_fixture_records()
    records[0]["target_move_uci"] = "a1h8"  # 시작 포지션에서 불가능한 수
    records[0].pop("next_fen", None)
    dataset_path = tmp_path / "corrupt.jsonl"
    _write_jsonl(dataset_path, records)
    with pytest.raises(ValueError, match="합법 수가 아니다"):
        build_dataset(
            DatasetBuildConfig(
                source="jsonl",
                dataset_path=str(dataset_path),
                split="all",
                loading="lazy",
                validate_sample_rate=1.0,
            )
        )
    # rate=0이면 스캔 검증을 건너뛰므로 빌드는 통과해야 한다.
    dataset = build_dataset(
        DatasetBuildConfig(
            source="jsonl",
            dataset_path=str(dataset_path),
            split="all",
            loading="lazy",
            validate_sample_rate=0.0,
        )
    )
    assert len(dataset) == len(records)


def test_lazy_dataset_pickles_without_file_handle() -> None:
    dataset = LazyJsonlDataset(
        DatasetBuildConfig(source="jsonl", dataset_path=FIXTURE_PILOT_PATH, split="all", loading="lazy")
    )
    _ = dataset[0]  # 핸들을 연 뒤에도
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._handle is None
    assert restored[0].position_id == dataset[0].position_id


def test_collate_partial_is_picklable() -> None:
    collate = partial(collate_position_samples, concept_vocab=["check"], fen_max_length=None)
    buffer = io.BytesIO()
    pickle.dump(collate, buffer)
    assert buffer.tell() > 0


def test_lazy_dataloader_with_spawned_worker() -> None:
    dataset = LazyJsonlDataset(
        DatasetBuildConfig(source="jsonl", dataset_path=FIXTURE_PILOT_PATH, split="all", loading="lazy")
    )
    dataloader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        num_workers=1,
        collate_fn=partial(collate_position_samples, concept_vocab=["check"]),
    )
    batch = next(iter(dataloader))
    assert batch["board_planes"].shape[0] == 3


def test_vectorized_listwise_loss_matches_loop_reference() -> None:
    from modalchess.train.losses import _listwise_policy_loss, _listwise_policy_loss_loop

    dataset = build_dataset(
        DatasetBuildConfig(source="jsonl", dataset_path=FIXTURE_PILOT_PATH, split="all")
    )
    samples = [dataset[i] for i in range(len(dataset))]
    batch = collate_position_samples(samples, concept_vocab=["check"])
    batch_size = len(samples)
    generator = torch.Generator().manual_seed(11)
    outputs = {
        "src_logits": torch.randn(batch_size, 64, generator=generator),
        "dst_logits": torch.randn(batch_size, 64, generator=generator),
        "promo_logits": torch.randn(batch_size, 5, generator=generator),
        "pair_logits": torch.randn(batch_size, 64, 64, generator=generator),
    }
    vectorized = _listwise_policy_loss(outputs, batch)
    loop = _listwise_policy_loss_loop(outputs, batch)
    assert torch.allclose(vectorized, loop, atol=1e-6)

    # 일부 샘플의 target을 결측(-100)으로 만들어도 동일해야 한다.
    batch["target_legal_move_index"] = batch["target_legal_move_index"].clone()
    batch["target_legal_move_index"][0] = -100
    vectorized_partial = _listwise_policy_loss(outputs, batch)
    loop_partial = _listwise_policy_loss_loop(outputs, batch)
    assert torch.allclose(vectorized_partial, loop_partial, atol=1e-6)

    # pair scorer가 없는 출력에서도 동일해야 한다.
    outputs_no_pair = {key: value for key, value in outputs.items() if key != "pair_logits"}
    assert torch.allclose(
        _listwise_policy_loss(outputs_no_pair, batch),
        _listwise_policy_loss_loop(outputs_no_pair, batch),
        atol=1e-6,
    )


def test_vectorized_listwise_loss_computes_in_fp32_under_low_precision_inputs() -> None:
    """bf16 입력이 들어와도 listwise 점수 합산은 fp32로 수행돼야 한다."""
    from modalchess.train.losses import _listwise_policy_loss

    dataset = build_dataset(
        DatasetBuildConfig(source="jsonl", dataset_path=FIXTURE_PILOT_PATH, split="all")
    )
    samples = [dataset[i] for i in range(len(dataset))]
    batch = collate_position_samples(samples, concept_vocab=["check"])
    batch_size = len(samples)
    generator = torch.Generator().manual_seed(3)
    outputs_bf16 = {
        "src_logits": torch.randn(batch_size, 64, generator=generator).bfloat16(),
        "dst_logits": torch.randn(batch_size, 64, generator=generator).bfloat16(),
        "promo_logits": torch.randn(batch_size, 5, generator=generator).bfloat16(),
        "pair_logits": torch.randn(batch_size, 64, 64, generator=generator).bfloat16(),
    }
    loss = _listwise_policy_loss(outputs_bf16, batch)
    assert loss.dtype == torch.float32
    outputs_fp32 = {key: value.float() for key, value in outputs_bf16.items()}
    loss_fp32 = _listwise_policy_loss(outputs_fp32, batch)
    assert torch.allclose(loss, loss_fp32, atol=1e-6)


def test_metrics_and_loss_survive_pin_memory_tuple_conversion() -> None:
    """torch의 pin_memory 재귀는 batch 내 tuple을 list로 바꾼다 — 소비처가 견뎌야 한다."""
    from modalchess.eval.metrics_move_quality import collect_move_prediction_rows
    from modalchess.train.losses import _listwise_policy_loss_loop

    dataset = build_dataset(
        DatasetBuildConfig(source="jsonl", dataset_path=FIXTURE_PILOT_PATH, split="all")
    )
    samples = [dataset[i] for i in range(len(dataset))]
    batch = collate_position_samples(samples, concept_vocab=["check"])
    # pin_memory와 동일한 변환을 명시적으로 재현: tuple → list
    batch["legal_moves_factorized"] = [
        [list(move) for move in moves] for moves in batch["legal_moves_factorized"]
    ]
    batch_size = len(samples)
    generator = torch.Generator().manual_seed(7)
    outputs = {
        "src_logits": torch.randn(batch_size, 64, generator=generator),
        "dst_logits": torch.randn(batch_size, 64, generator=generator),
        "promo_logits": torch.randn(batch_size, 5, generator=generator),
        "pair_logits": torch.randn(batch_size, 64, 64, generator=generator),
    }
    rows = collect_move_prediction_rows(outputs, batch, topk=[1, 3])
    assert len(rows) == sum(1 for s in samples if s.target_move_uci is not None)
    loss = _listwise_policy_loss_loop(outputs, batch)
    assert torch.isfinite(loss)


def test_warmup_cosine_schedule_shape() -> None:
    total_steps = 100
    warmup_steps = 10
    values = [
        warmup_cosine_lr_lambda(
            step, total_steps=total_steps, warmup_steps=warmup_steps, min_lr_ratio=0.1
        )
        for step in range(total_steps)
    ]
    assert values[0] == pytest.approx(1 / warmup_steps)
    assert values[warmup_steps - 1] == pytest.approx(1.0)
    assert max(values) == pytest.approx(1.0)
    assert values[-1] >= 0.1
    assert values[-1] == pytest.approx(0.1, abs=0.01)
    # warmup 이후 단조 감소
    post_warmup = values[warmup_steps:]
    assert all(a >= b for a, b in zip(post_warmup, post_warmup[1:]))


def test_build_lr_scheduler_constant_returns_none() -> None:
    model = torch.nn.Linear(4, 4)
    optimizer = build_optimizer(model, learning_rate=0.1, weight_decay=0.0)
    assert build_lr_scheduler(optimizer, None, total_steps=10) is None
    assert build_lr_scheduler(optimizer, {"name": "constant"}, total_steps=10) is None
    scheduler = build_lr_scheduler(
        optimizer, {"name": "warmup_cosine", "warmup_ratio": 0.1}, total_steps=10
    )
    assert scheduler is not None
    with pytest.raises(ValueError, match="지원하지 않는"):
        build_lr_scheduler(optimizer, {"name": "unknown"}, total_steps=10)


def test_min_rating_filter_drops_low_and_missing_rating_games() -> None:
    pgn_text = """[Event "Rated Blitz game"]
[Site "https://lichess.org/high0001"]
[White "a"]
[Black "b"]
[WhiteElo "2000"]
[BlackElo "1900"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 *

[Event "Rated Blitz game"]
[Site "https://lichess.org/low00001"]
[White "c"]
[Black "d"]
[WhiteElo "1500"]
[BlackElo "1900"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O *

[Event "Rated Blitz game"]
[Site "https://lichess.org/miss0001"]
[White "e"]
[Black "f"]

1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 4. g3 d5 5. cxd5 Nxd5 *
"""
    pgn_path = Path("outputs") / "_test_min_rating.pgn"
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    pgn_path.write_text(pgn_text, encoding="utf-8")
    try:
        records_by_split, report = build_supervised_records_from_pgn(
            [pgn_path],
            PgnPilotBuildConfig(min_rating=1800, min_game_plies=1),
        )
        assert report["games_kept"] == 1
        assert report["drop_reasons"].get("below_min_rating") == 1
        assert report["drop_reasons"].get("missing_rating") == 1
        total_records = sum(len(records) for records in records_by_split.values())
        assert total_records == 10
    finally:
        pgn_path.unlink(missing_ok=True)


def test_rated_only_rejects_unrated_event_and_explicit_false_header(tmp_path: Path) -> None:
    pgn_text = """[Event "Unrated Blitz game"]
[Site "https://lichess.org/unrated1"]

1. e4 e5 *

[Event "Rated Blitz game"]
[Site "https://lichess.org/false001"]
[Rated "False"]

1. d4 d5 *

[Event "Rated Blitz game"]
[Site "https://lichess.org/rated001"]

1. c4 e5 *
"""
    pgn_path = tmp_path / "rated_only.pgn"
    pgn_path.write_text(pgn_text, encoding="utf-8")

    records_by_split, report = build_supervised_records_from_pgn(
        [pgn_path],
        PgnPilotBuildConfig(rated_only=True, min_game_plies=1),
    )

    assert report["games_seen"] == 3
    assert report["games_kept"] == 1
    assert report["drop_reasons"].get("unrated_game") == 2
    assert sum(len(records) for records in records_by_split.values()) == 2


def test_run_training_early_stops_with_frozen_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(train_entry, "resolve_device", lambda: torch.device("cpu"))
    config = {
        "seed": 11,
        "output_dir": str(tmp_path / "early_stop_run"),
        "model_config_paths": [
            "configs/model/board_encoder.yaml",
            "configs/model/heads.yaml",
        ],
        "model": {"d_model": 32, "num_layers": 1, "num_heads": 2, "dropout": 0.0},
        "train_dataset": {
            "source": "jsonl",
            "dataset_path": FIXTURE_PILOT_PATH,
            "split": "train",
        },
        "val_dataset": {
            "source": "jsonl",
            "dataset_path": FIXTURE_PILOT_PATH,
            "split": "val",
        },
        "train": {
            "batch_size": 4,
            "eval_batch_size": 4,
            "epochs": 5,
            "learning_rate": 0.0,
            "weight_decay": 0.0,
            "early_stop_patience": 1,
        },
        "losses": {"policy": 1.0, "state_probe": 1.0},
    }
    metrics = train_entry.run_training(config)
    # lr=0이라 val NLL이 개선되지 않으므로 epoch 2에서 patience=1로 중단해야 한다.
    assert metrics["early_stopped_at_epoch"] == 2
    assert len(metrics["epoch_metrics"]) == 2
    assert metrics["best_epoch"] == 1
    train_metrics = metrics["epoch_metrics"][0]["train"]
    assert "samples_per_second" in train_metrics
    assert "learning_rate" in train_metrics


def test_run_training_with_lr_schedule_records_decaying_lr(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(train_entry, "resolve_device", lambda: torch.device("cpu"))
    config = {
        "seed": 11,
        "output_dir": str(tmp_path / "schedule_run"),
        "model_config_paths": [
            "configs/model/board_encoder.yaml",
            "configs/model/heads.yaml",
        ],
        "model": {"d_model": 32, "num_layers": 1, "num_heads": 2, "dropout": 0.0},
        "train_dataset": {
            "source": "jsonl",
            "dataset_path": FIXTURE_PILOT_PATH,
            "split": "train",
        },
        "val_dataset": {
            "source": "jsonl",
            "dataset_path": FIXTURE_PILOT_PATH,
            "split": "val",
        },
        "train": {
            "batch_size": 4,
            "eval_batch_size": 4,
            "epochs": 3,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "lr_schedule": {"name": "warmup_cosine", "warmup_ratio": 0.2, "min_lr_ratio": 0.1},
        },
        "losses": {"policy": 1.0, "state_probe": 1.0},
    }
    metrics = train_entry.run_training(config)
    lrs = [record["train"]["learning_rate"] for record in metrics["epoch_metrics"]]
    assert len(lrs) == 3
    # cosine 구간에서 LR이 감소해야 한다.
    assert lrs[-1] < lrs[0]
    assert lrs[-1] >= 0.001 * 0.1 - 1e-9
