"""Frozen 문장 인코더(MiniLM mean-pool) 임베딩 계산·캐시.

두 인코더가 frozen이므로 텍스트 임베딩은 corpus당 1회만 계산해 .pt로 캐시한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def normalize_comment(text: str | None) -> str:
    """multi-positive/ignore group 키로 쓰는 정규화 텍스트."""
    return " ".join(str(text or "").lower().split())


def encode_texts(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    max_length: int = 128,
    device: torch.device | None = None,
) -> torch.Tensor:
    """mean-pooled, L2-normalized 문장 임베딩 [N, dim]을 반환한다."""
    from transformers import AutoModel, AutoTokenizer

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(target_device)
    outputs: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(target_device)
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        outputs.append(torch.nn.functional.normalize(pooled, dim=1).cpu())
    return torch.cat(outputs, dim=0)


def precompute_corpus_text_embeddings(
    corpus_root: str | Path,
    output_root: str | Path,
    family: str = "annotated_sidecar",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    splits: tuple[str, ...] = ("train", "val", "test"),
    batch_size: int = 256,
) -> dict[str, str]:
    """corpus의 comment_text를 인코딩해 split별 {probe_id, embedding, position_id, ...}를 저장."""
    corpus_path = Path(corpus_root)
    out_path = Path(output_root)
    out_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for split in splits:
        rows = [
            json.loads(line)
            for line in (corpus_path / f"{family}_{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        texts = [str(row.get("comment_text") or "") for row in rows]
        embeddings = encode_texts(texts, model_name=model_name, batch_size=batch_size)
        payload = {
            "model_name": model_name,
            "probe_id": [str(row["probe_id"]) for row in rows],
            "position_id": [str(row.get("position_id")) for row in rows],
            "source_family": [str(row.get("source_family") or "unknown") for row in rows],
            "normalized_text": [normalize_comment(row.get("comment_text")) for row in rows],
            "embedding": embeddings,
        }
        target = out_path / f"{family}_{split}_text.pt"
        torch.save(payload, target)
        written[split] = str(target)
    return written
