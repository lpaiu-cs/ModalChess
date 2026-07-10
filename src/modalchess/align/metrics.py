"""Connector retrieval 지표(strict tie R@k/MRR) + global·within-family permutation null.

strict tie 의미는 gate2/raw_text_retrieval과 동일하게 맞춘다:
rank = (#strictly-greater) + (#tied incl self). 비교 가능성 유지.
"""

from __future__ import annotations

import random

import torch


def _ranks_strict(scores: torch.Tensor, target_indices: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-6) -> torch.Tensor:
    n = scores.size(0)
    rows = torch.arange(n)
    target_scores = scores[rows, target_indices]
    greater = (scores > target_scores.unsqueeze(1)).sum(dim=1)
    tie = torch.isclose(scores, target_scores.unsqueeze(1), atol=atol, rtol=rtol).sum(dim=1)
    return greater + tie  # tied(자기 포함)를 더해 conservative rank


def retrieval_metrics(
    query: torch.Tensor,
    gallery: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10, 50),
    target_indices: torch.Tensor | None = None,
) -> dict[str, float]:
    """query[i]가 gallery에서 target[i](기본 대각)를 얼마나 잘 찾는지. strict tie."""
    n = query.size(0)
    if target_indices is None:
        target_indices = torch.arange(n)
    scores = query @ gallery.transpose(0, 1)
    ranks = _ranks_strict(scores, target_indices).float()
    out: dict[str, float] = {"mrr": float((1.0 / ranks).mean())}
    for k in ks:
        out[f"r@{k}"] = float((ranks <= k).float().mean())
    return out


def bidirectional_metrics(board_proj: torch.Tensor, text_proj: torch.Tensor, ks=(1, 5, 10, 50)) -> dict[str, dict[str, float]]:
    return {
        "board_to_text": retrieval_metrics(board_proj, text_proj, ks),
        "text_to_board": retrieval_metrics(text_proj, board_proj, ks),
    }


def _global_derangement(n: int, rng: random.Random) -> list[int]:
    perm = list(range(n))
    for _ in range(64):
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm
    # fallback: 회전(고정점 없음 보장)
    return [(i + 1) % n for i in range(n)]


def _within_family_derangement(source_family: list[str], rng: random.Random) -> tuple[list[int], list[int]]:
    """같은 family 내에서만 섞은 순열. 반환: (perm, coverable_indices).

    family 크기 1은 섞을 수 없어 자기 자신으로 남긴다(coverable에서 제외).
    """
    from collections import defaultdict

    groups: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(source_family):
        groups[family].append(index)
    perm = list(range(len(source_family)))
    coverable: list[int] = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        shuffled = idxs[:]
        for _ in range(64):
            rng.shuffle(shuffled)
            if all(shuffled[k] != idxs[k] for k in range(len(idxs))):
                break
        else:
            shuffled = idxs[1:] + idxs[:1]
        for original, replacement in zip(idxs, shuffled):
            perm[original] = replacement
        coverable.extend(idxs)
    return perm, sorted(coverable)


def null_control(
    board_proj: torch.Tensor,
    text_proj: torch.Tensor,
    source_family: list[str],
    repeats: int = 50,
    seed: int = 20260710,
    ks: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, dict[str, float]]:
    """real vs global-null vs within-family-null (text→board, board→text 모두).

    within-family null은 family 크기>=2인 query만 대상으로 계산(coverable subset).
    """
    rng = random.Random(seed)
    n = board_proj.size(0)
    real = bidirectional_metrics(board_proj, text_proj, ks)

    def run_null(perm_fn, subset: list[int] | None):
        b2t_mrr, t2b_mrr = [], []
        for r in range(repeats):
            perm = perm_fn(random.Random(seed + r))
            if isinstance(perm, tuple):
                perm, cover = perm
            else:
                cover = None
            perm_t = torch.tensor(perm, dtype=torch.long)
            shuffled_text = text_proj.index_select(0, perm_t)
            if cover is not None and subset is not None:
                idx = torch.tensor(subset, dtype=torch.long)
                bm = retrieval_metrics(board_proj.index_select(0, idx), shuffled_text, ks, target_indices=idx)
                tm = retrieval_metrics(shuffled_text.index_select(0, idx), board_proj, ks, target_indices=idx)
            else:
                bm = retrieval_metrics(board_proj, shuffled_text, ks)
                tm = retrieval_metrics(shuffled_text, board_proj, ks)
            b2t_mrr.append(bm["mrr"]); t2b_mrr.append(tm["mrr"])
        import statistics as st
        return {
            "board_to_text_mrr_mean": st.mean(b2t_mrr), "board_to_text_mrr_max": max(b2t_mrr),
            "text_to_board_mrr_mean": st.mean(t2b_mrr), "text_to_board_mrr_max": max(t2b_mrr),
        }

    global_null = run_null(lambda r: _global_derangement(n, r), None)
    _, coverable = _within_family_derangement(source_family, random.Random(seed))
    within_null = run_null(lambda r: _within_family_derangement(source_family, r), coverable)
    within_null["coverable_fraction"] = len(coverable) / max(n, 1)
    return {"real": real, "global_null": global_null, "within_family_null": within_null}
