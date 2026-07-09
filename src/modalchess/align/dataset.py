"""Precompute된 board/text 임베딩을 probe_id로 정렬한 pair 데이터 + family_blocked sampler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch

from modalchess.align.text_embed import normalize_comment


@dataclass(slots=True)
class AlignedPairs:
    board: torch.Tensor            # [N, board_dim]
    text: torch.Tensor            # [N, text_dim]
    probe_id: list[str]
    position_id: list[str]
    source_family: list[str]
    normalized_text: list[str]

    def __len__(self) -> int:
        return self.board.size(0)

    def subset(self, indices: list[int]) -> "AlignedPairs":
        index = torch.tensor(indices, dtype=torch.long)
        return AlignedPairs(
            board=self.board.index_select(0, index),
            text=self.text.index_select(0, index),
            probe_id=[self.probe_id[i] for i in indices],
            position_id=[self.position_id[i] for i in indices],
            source_family=[self.source_family[i] for i in indices],
            normalized_text=[self.normalized_text[i] for i in indices],
        )


def load_aligned_pairs(
    board_embedding_path: str | Path,
    text_embedding_path: str | Path,
    pool: str = "board_pooled",
) -> AlignedPairs:
    """board .pt(pool 선택)와 text .pt를 공통 probe_id 교집합으로 정렬한다."""
    board_payload = torch.load(board_embedding_path, map_location="cpu", weights_only=False)
    text_payload = torch.load(text_embedding_path, map_location="cpu", weights_only=False)
    board_index = {str(pid): i for i, pid in enumerate(board_payload["probe_id"])}
    text_index = {str(pid): i for i, pid in enumerate(text_payload["probe_id"])}
    common = [pid for pid in text_payload["probe_id"] if str(pid) in board_index]
    common = [str(pid) for pid in common]
    if not common:
        raise ValueError("board/text 임베딩에 공통 probe_id가 없다.")
    board_order = torch.tensor([board_index[pid] for pid in common], dtype=torch.long)
    text_order = torch.tensor([text_index[pid] for pid in common], dtype=torch.long)
    board = board_payload[pool].index_select(0, board_order).float()
    text = text_payload["embedding"].index_select(0, text_order).float()
    # 메타데이터는 text payload 우선(정규화 텍스트 포함), 없으면 유도
    text_meta_index = {str(pid): i for i, pid in enumerate(text_payload["probe_id"])}
    def meta(field: str, default_from_board: bool = False) -> list[str]:
        if field in text_payload:
            return [str(text_payload[field][text_meta_index[pid]]) for pid in common]
        if default_from_board and field in board_payload:
            return [str(board_payload[field][board_index[pid]]) for pid in common]
        return ["unknown"] * len(common)
    position_id = meta("position_id", default_from_board=True)
    source_family = meta("source_family")
    if "normalized_text" in text_payload:
        normalized_text = meta("normalized_text")
    else:
        normalized_text = [normalize_comment("") for _ in common]
    return AlignedPairs(
        board=board,
        text=text,
        probe_id=common,
        position_id=position_id,
        source_family=source_family,
        normalized_text=normalized_text,
    )


class FamilyBlockedSampler:
    """batch = F families × m samples(m>=2)로 within-family hard negatives를 보장하는 sampler.

    - `min_family_size`(=m) 미만 family는 blocked 대상에서 빼고 misc pool로 별도 채운다(전량 버리지 않음).
    - family/샘플 순서를 seed로 셔플해 epoch마다 다르게 구성.
    """

    def __init__(
        self,
        source_family: list[str],
        families_per_batch: int,
        samples_per_family: int,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if samples_per_family < 2:
            raise ValueError("family_blocked에는 samples_per_family >= 2가 필요하다.")
        self.families_per_batch = families_per_batch
        self.samples_per_family = samples_per_family
        self.batch_size = families_per_batch * samples_per_family
        self.drop_last = drop_last
        self.rng = random.Random(seed)
        self.by_family: dict[str, list[int]] = {}
        for index, family in enumerate(source_family):
            self.by_family.setdefault(family, []).append(index)
        self.blockable = [f for f, idxs in self.by_family.items() if len(idxs) >= samples_per_family]
        self.misc: list[int] = []
        for family, idxs in self.by_family.items():
            if len(idxs) < samples_per_family:
                self.misc.extend(idxs)
        self._num_batches = self._estimate_num_batches()

    def _estimate_num_batches(self) -> int:
        total_blockable = sum(len(self.by_family[f]) for f in self.blockable)
        return max(1, total_blockable // self.batch_size)

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        # family별 셔플된 큐를 만들고, 매 batch마다 F family에서 m개씩 뽑는다.
        pools = {f: self.rng.sample(self.by_family[f], len(self.by_family[f])) for f in self.blockable}
        misc_pool = self.rng.sample(self.misc, len(self.misc)) if self.misc else []
        active = [f for f in self.blockable if len(pools[f]) >= self.samples_per_family]
        self.rng.shuffle(active)
        produced = 0
        while produced < self._num_batches:
            ready = [f for f in active if len(pools[f]) >= self.samples_per_family]
            if len(ready) < self.families_per_batch:
                break
            chosen = self.rng.sample(ready, self.families_per_batch)
            batch: list[int] = []
            for family in chosen:
                for _ in range(self.samples_per_family):
                    batch.append(pools[family].pop())
            self.rng.shuffle(batch)
            yield batch
            produced += 1


def build_batch_masks(pairs: AlignedPairs, indices: list[int]):
    """batch 인덱스에 대한 positive/ignore mask (connector.build_pair_masks 래퍼)."""
    from modalchess.align.connector import build_pair_masks

    position_ids = [pairs.position_id[i] for i in indices]
    normalized_texts = [pairs.normalized_text[i] for i in indices]
    return build_pair_masks(position_ids, normalized_texts)
