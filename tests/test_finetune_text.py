"""finetune_text의 순수 로직(mean-pool, split 정렬) 테스트 — HF 모델 로드 없이."""

from __future__ import annotations

import json

import torch

from modalchess.align.finetune_text import load_finetune_split, mean_pool


def test_mean_pool_matches_frozen_pipeline() -> None:
    hidden = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],  # 마지막 토큰은 패딩
    ])
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0]]))


def test_mean_pool_all_padding_is_safe() -> None:
    hidden = torch.ones((1, 2, 4))
    mask = torch.zeros((1, 2))
    pooled = mean_pool(hidden, mask)
    assert torch.isfinite(pooled).all()


def test_load_finetune_split_alignment(tmp_path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"probe_id": "p1", "comment_text": "first", "source_family": "famA", "position_id": "pos1"},
        {"probe_id": "p2", "comment_text": "second", "source_family": "famB", "position_id": "pos2"},
        {"probe_id": "p3", "comment_text": "third", "source_family": "famA", "position_id": "pos3"},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    board_path = tmp_path / "board.pt"
    torch.save({
        "probe_id": ["p3", "p1"],  # 순서가 다르고 p2는 없음
        "board_pooled": torch.tensor([[3.0, 3.0], [1.0, 1.0]]),
    }, board_path)
    features_path = tmp_path / "features.pt"
    torch.save({
        "probe_id": ["p1", "p2", "p3"],
        "board_features": torch.tensor([[10.0], [20.0], [30.0]]),
        "text_features": torch.tensor([[0.1], [0.2], [0.3]]),
    }, features_path)

    split = load_finetune_split(corpus, board_path, features_path)
    # board .pt 순서 기준, 교집합 {p3, p1}
    assert split["probe_id"] == ["p3", "p1"]
    assert split["texts"] == ["third", "first"]
    # board = [임베딩(2) | board_feat(1)] concat
    assert split["board"].shape == (2, 3)
    assert torch.allclose(split["board"][0], torch.tensor([3.0, 3.0, 30.0]))
    assert torch.allclose(split["text_features"][1], torch.tensor([0.1]))
    assert split["source_family"] == ["famA", "famA"]
    assert split["normalized_text"] == ["third", "first"]
