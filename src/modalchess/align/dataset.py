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
    features_path: str | Path | None = None,
    feature_mode: str = "hybrid",
) -> AlignedPairs:
    """board .pt(pool 선택)와 text .pt를 공통 probe_id 교집합으로 정렬한다.

    features_path가 주어지면 symbolic_features .pt(probe_id 키)를 결합한다:
    - feature_mode="hybrid": 임베딩에 심볼릭 특징을 concat (기본).
    - feature_mode="symbolic_only": 임베딩 대신 심볼릭 특징만 사용 (control).
    """
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
    if features_path is not None:
        feature_payload = torch.load(features_path, map_location="cpu", weights_only=False)
        feature_index = {str(pid): i for i, pid in enumerate(feature_payload["probe_id"])}
        missing = [pid for pid in common if pid not in feature_index]
        if missing:
            raise ValueError(f"symbolic features에 없는 probe_id {len(missing)}개 (예: {missing[:3]})")
        feature_order = torch.tensor([feature_index[pid] for pid in common], dtype=torch.long)
        board_feat = feature_payload["board_features"].index_select(0, feature_order).float()
        text_feat = feature_payload["text_features"].index_select(0, feature_order).float()
        if feature_mode == "hybrid":
            board = torch.cat([board, board_feat], dim=1)
            text = torch.cat([text, text_feat], dim=1)
        elif feature_mode == "symbolic_only":
            board = board_feat
            text = text_feat
        else:
            raise ValueError(f"unknown feature_mode: {feature_mode}")
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
        drop_last: bool = False,
    ) -> None:
        if families_per_batch < 1:
            raise ValueError("families_per_batch >= 1이 필요하다.")
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
        if not self.blockable:
            raise ValueError(
                "family_blocked batch를 만들 수 없다: "
                f"samples_per_family={samples_per_family} 이상인 source_family가 필요하다."
            )
        self._num_batches = self._estimate_num_batches()

    def _estimate_num_batches(self) -> int:
        groups_per_family = [
            len(self.by_family[family]) // self.samples_per_family
            for family in self.blockable
        ]
        misc_count = len(self.misc) + sum(
            len(self.by_family[family]) % self.samples_per_family
            for family in self.blockable
        )
        batches = 0
        while any(count > 0 for count in groups_per_family):
            family_slots = self.families_per_batch
            if misc_count > 0 and family_slots > 1:
                family_slots -= 1
            ready = sorted(
                (index for index, count in enumerate(groups_per_family) if count > 0),
                key=lambda index: groups_per_family[index],
                reverse=True,
            )
            chosen = ready[:family_slots]
            for index in chosen:
                groups_per_family[index] -= 1
            remaining = self.batch_size - len(chosen) * self.samples_per_family
            misc_used = min(remaining, misc_count)
            misc_count -= misc_used
            batch_size = len(chosen) * self.samples_per_family + misc_used
            if not self.drop_last or batch_size == self.batch_size:
                batches += 1
        return batches

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        # family별 셔플된 큐를 만들고, 매 batch마다 F family에서 m개씩 뽑는다.
        pools: dict[str, list[int]] = {}
        misc_pool = self.rng.sample(self.misc, len(self.misc)) if self.misc else []
        for family in self.blockable:
            shuffled = self.rng.sample(self.by_family[family], len(self.by_family[family]))
            blockable_count = (len(shuffled) // self.samples_per_family) * self.samples_per_family
            pools[family] = shuffled[:blockable_count]
            misc_pool.extend(shuffled[blockable_count:])
        self.rng.shuffle(misc_pool)
        active = [f for f in self.blockable if len(pools[f]) >= self.samples_per_family]
        while active:
            ready = [f for f in active if len(pools[f]) >= self.samples_per_family]
            if not ready:
                break
            # misc tail이 있으면 한 family slot을 비워 실제 batch에 섞는다. 단, 최소 한
            # blockable family는 유지해 within-family hard negative 계약을 보존한다.
            family_slots = self.families_per_batch
            if misc_pool and family_slots > 1:
                family_slots -= 1
            self.rng.shuffle(ready)
            ready.sort(key=lambda family: len(pools[family]), reverse=True)
            chosen = ready[:family_slots]
            batch: list[int] = []
            for family in chosen:
                for _ in range(self.samples_per_family):
                    batch.append(pools[family].pop())
            remaining = self.batch_size - len(batch)
            for _ in range(min(remaining, len(misc_pool))):
                batch.append(misc_pool.pop())
            self.rng.shuffle(batch)
            active = [f for f in active if len(pools[f]) >= self.samples_per_family]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch


def build_batch_masks(pairs: AlignedPairs, indices: list[int]):
    """batch 인덱스에 대한 positive/ignore mask (connector.build_pair_masks 래퍼)."""
    from modalchess.align.connector import build_pair_masks

    position_ids = [pairs.position_id[i] for i in indices]
    normalized_texts = [pairs.normalized_text[i] for i in indices]
    return build_pair_masks(position_ids, normalized_texts)
