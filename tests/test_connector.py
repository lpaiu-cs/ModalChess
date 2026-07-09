"""alignment connector 코어(loss/mask/heads) 정합성 테스트."""

from __future__ import annotations

import torch

from modalchess.align.connector import (
    AlignmentConnector,
    ConnectorConfig,
    build_pair_masks,
    multi_positive_infonce,
)


def _diag_masks(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.eye(n, dtype=torch.bool)
    ignore = torch.zeros((n, n), dtype=torch.bool)
    return pos, ignore


def test_infonce_recovers_diagonal_on_separable_pairs() -> None:
    torch.manual_seed(0)
    n, d = 32, 16
    base = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    pos, ignore = _diag_masks(n)
    # board==text (완전 정렬) → 큰 scale에서 loss가 0에 수렴해야 한다.
    loss = multi_positive_infonce(base, base, torch.tensor(30.0), pos, ignore)
    assert loss.item() < 1e-2
    # 랜덤 무정렬은 loss가 크다.
    other = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    loss_rand = multi_positive_infonce(base, other, torch.tensor(30.0), pos, ignore)
    assert loss_rand.item() > loss.item() + 1.0


def test_infonce_symmetric_and_nonnegative() -> None:
    torch.manual_seed(1)
    n, d = 16, 8
    b = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    t = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    pos, ignore = _diag_masks(n)
    loss = multi_positive_infonce(b, t, torch.tensor(10.0), pos, ignore)
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)


def test_ignore_mask_excludes_false_negative() -> None:
    # 행 0과 1이 같은 텍스트(다른 board)라 서로 negative로 세면 안 된다.
    torch.manual_seed(2)
    n, d = 6, 8
    b = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    t = b.clone()  # 완전 정렬
    # 행 0,1의 텍스트를 동일하게(같은 벡터) 만들어 충돌 유발
    t[1] = t[0]
    pos = torch.eye(n, dtype=torch.bool)
    ignore = torch.zeros((n, n), dtype=torch.bool)
    ignore[0, 1] = ignore[1, 0] = True
    loss_masked = multi_positive_infonce(b, t, torch.tensor(30.0), pos, ignore)
    # ignore 없이 계산하면 0/1이 서로 강한 negative가 되어 loss가 더 커진다.
    loss_unmasked = multi_positive_infonce(b, t, torch.tensor(30.0), pos, torch.zeros_like(ignore))
    assert loss_masked.item() < loss_unmasked.item()


def test_multi_positive_same_board_counts_as_positive() -> None:
    # 같은 position_id 두 행이 서로 positive면, 그 board가 두 코멘트 중 하나만 맞춰도 낮은 loss.
    torch.manual_seed(3)
    n, d = 4, 8
    b = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    t = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    # 행 0,1을 같은 board로: board 벡터 동일, text는 다름
    b[1] = b[0]
    pos = torch.eye(n, dtype=torch.bool)
    pos[0, 1] = pos[1, 0] = True
    ignore = torch.zeros((n, n), dtype=torch.bool)
    loss = multi_positive_infonce(b, t, torch.tensor(10.0), pos, ignore)
    assert torch.isfinite(loss) and loss.item() >= 0.0


def test_build_pair_masks_from_metadata() -> None:
    position_ids = ["p0", "p1", "p1", "p3"]  # 1,2는 같은 board
    normalized_texts = ["a", "b", "c", "a"]  # 0,3은 같은 텍스트
    pos, ignore = build_pair_masks(position_ids, normalized_texts)
    assert pos[1, 2] and pos[2, 1] and pos[0, 0]
    assert ignore[0, 3] and ignore[3, 0]
    assert not ignore[1, 2]  # positive는 ignore가 아니다
    assert not pos[0, 3]


def test_connector_heads_shapes_and_norm() -> None:
    for projection in ("linear", "mlp"):
        cfg = ConnectorConfig(board_dim=384, text_dim=384, proj_dim=128, projection=projection)
        model = AlignmentConnector(cfg)
        board = torch.randn(5, 384)
        text = torch.randn(5, 384)
        zb, zt = model(board, text)
        assert zb.shape == (5, 128) and zt.shape == (5, 128)
        assert torch.allclose(zb.norm(dim=1), torch.ones(5), atol=1e-5)
        assert torch.allclose(zt.norm(dim=1), torch.ones(5), atol=1e-5)
        assert float(model.scale().detach()) > 1.0


def test_family_blocked_sampler_guarantees_within_family_negatives() -> None:
    from collections import Counter

    from modalchess.align.dataset import FamilyBlockedSampler

    # 3 family × 넉넉한 샘플 + singleton tail
    families = (["A"] * 20) + (["B"] * 20) + (["C"] * 20) + ["D", "E"]
    sampler = FamilyBlockedSampler(
        families, families_per_batch=2, samples_per_family=4, seed=1
    )
    assert set(sampler.blockable) == {"A", "B", "C"}
    assert set(sampler.misc) == {60, 61}  # D,E 인덱스
    batches = list(iter(sampler))
    assert len(batches) >= 1
    for batch in batches:
        assert len(batch) == 8  # 2 family × 4
        fam_counts = Counter(families[i] for i in batch)
        # 각 뽑힌 family는 정확히 m개 → within-family negative 보장
        assert all(c == 4 for c in fam_counts.values())
        assert len(fam_counts) == 2


def test_build_batch_masks_wrapper() -> None:
    import torch as T

    from modalchess.align.dataset import AlignedPairs, build_batch_masks

    pairs = AlignedPairs(
        board=T.randn(4, 8), text=T.randn(4, 8),
        probe_id=["a", "b", "c", "d"],
        position_id=["p0", "p0", "p2", "p3"],
        normalized_text=["x", "y", "z", "x"],
        source_family=["f", "f", "g", "g"],
    )
    pos, ignore = build_batch_masks(pairs, [0, 1, 2, 3])
    assert pos[0, 1] and pos[1, 0]        # 같은 position → multi-positive
    assert ignore[0, 3] and ignore[3, 0]  # 같은 텍스트, 다른 position → ignore


def test_connector_trains_toward_alignment() -> None:
    # 작은 합성 정렬 문제에서 몇 스텝만에 retrieval R@1이 오른다.
    torch.manual_seed(4)
    n, d = 64, 32
    latent = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    board = latent + 0.05 * torch.randn(n, d)
    text = latent + 0.05 * torch.randn(n, d)
    cfg = ConnectorConfig(board_dim=d, text_dim=d, proj_dim=16, projection="mlp", dropout=0.0)
    model = AlignmentConnector(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    pos = torch.eye(n, dtype=torch.bool)
    ignore = torch.zeros((n, n), dtype=torch.bool)

    def recall_at_1() -> float:
        model.eval()
        with torch.no_grad():
            zb, zt = model(board, text)
            sim = zb @ zt.transpose(0, 1)
            pred = sim.argmax(dim=1)
        return float((pred == torch.arange(n)).float().mean())

    start = recall_at_1()
    for _ in range(100):
        model.train()
        opt.zero_grad()
        zb, zt = model(board, text)
        loss = multi_positive_infonce(zb, zt, model.scale(), pos, ignore)
        loss.backward()
        opt.step()
    assert recall_at_1() > max(start, 0.5)
