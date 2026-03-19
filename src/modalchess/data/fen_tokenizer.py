"""FEN 문자열용 경량 문자 단위 토크나이저."""

from __future__ import annotations

from dataclasses import dataclass

import torch


FEN_VOCAB = [
    "<pad>",
    "<unk>",
    " ",
    "/",
    "-",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "P",
    "N",
    "B",
    "R",
    "Q",
    "K",
    "p",
    "n",
    "b",
    "r",
    "q",
    "k",
    "w",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "K",
    "Q",
    "k",
    "q",
]


@dataclass(slots=True)
class FenTokenizer:
    """FEN 문자열을 토큰 ID와 attention mask로 변환한다."""

    vocab: list[str]
    pad_token_id: int = 0
    unk_token_id: int = 1

    @classmethod
    def default(cls) -> "FenTokenizer":
        """기본 FEN 문자 어휘로 토크나이저를 만든다."""
        deduped_vocab: list[str] = []
        seen: set[str] = set()
        for token in FEN_VOCAB:
            if token not in seen:
                deduped_vocab.append(token)
                seen.add(token)
        return cls(vocab=deduped_vocab)

    @property
    def token_to_id(self) -> dict[str, int]:
        """문자에서 ID로의 매핑을 반환한다."""
        return {token: idx for idx, token in enumerate(self.vocab)}

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        """단일 FEN 문자열을 문자 단위 ID 시퀀스로 인코딩한다."""
        token_to_id = self.token_to_id
        ids = [token_to_id.get(char, self.unk_token_id) for char in text]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def batch_encode(
        self,
        texts: list[str],
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """여러 FEN 문자열을 패딩된 토큰 텐서와 mask로 변환한다."""
        encoded = [self.encode(text, max_length=max_length) for text in texts]
        target_length = max(len(ids) for ids in encoded) if encoded else 0
        if max_length is not None:
            target_length = min(target_length, max_length)
        token_ids = torch.full((len(encoded), target_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), target_length), dtype=torch.bool)
        for row, ids in enumerate(encoded):
            ids = ids[:target_length]
            if ids:
                token_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[row, : len(ids)] = True
        return token_ids, attention_mask
