"""레버 ②b: text encoder(MiniLM) contrastive fine-tune — frozen-text 규율의 명시적 해제.

진단 ②(segment_diagnosis)가 비언급 세그먼트의 심볼릭 레버 소진을 확증한 뒤에만 쓰는 단계.
- board 쪽은 계속 frozen: precompute 임베딩(384) + 심볼릭 특징(140).
- text 쪽: MiniLM 전층 학습 → mean-pool → L2norm(frozen 파이프라인과 동일) → 심볼릭
  특징(333) concat → AlignmentConnector text head. 배칭·손실·평가 장치는 전부 동결 재사용.
- kill criteria(사전 명시): 비언급 세그먼트 t2b MRR < 1.3×(기준 0.0578) 또는
  within-family null 실패(style 과적합) → 세그먼트는 데이터 한계로 종결.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import torch

from modalchess.align.connector import (
    AlignmentConnector,
    ConnectorConfig,
    build_pair_masks,
    multi_positive_infonce,
)
from modalchess.align.dataset import FamilyBlockedSampler
from modalchess.align.metrics import bidirectional_metrics, null_control
from modalchess.align.text_embed import normalize_comment

# hybrid p128 seed11(fixed sampler)의 비언급 세그먼트 실측 — 판정 기준선
BASELINE_SEGMENT_T2B_MRR = 0.0578


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """frozen 파이프라인(text_embed.encode_texts)과 동일한 mean-pool."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)


def load_finetune_split(
    corpus_path: str | Path,
    board_embedding_path: str | Path,
    features_path: str | Path,
    pool: str = "board_pooled",
) -> dict[str, Any]:
    """corpus(raw text) + frozen board 임베딩 + 심볼릭 특징을 probe_id로 정렬."""
    board_payload = torch.load(board_embedding_path, map_location="cpu", weights_only=False)
    feature_payload = torch.load(features_path, map_location="cpu", weights_only=False)
    rows_by_probe: dict[str, dict[str, Any]] = {}
    with open(corpus_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows_by_probe[str(row["probe_id"])] = row
    board_index = {str(pid): i for i, pid in enumerate(board_payload["probe_id"])}
    feature_index = {str(pid): i for i, pid in enumerate(feature_payload["probe_id"])}
    common = [pid for pid in board_payload["probe_id"] if str(pid) in rows_by_probe and str(pid) in feature_index]
    common = [str(pid) for pid in common]
    if not common:
        raise ValueError("corpus/board/features 공통 probe_id가 없다.")
    board_order = torch.tensor([board_index[pid] for pid in common], dtype=torch.long)
    feat_order = torch.tensor([feature_index[pid] for pid in common], dtype=torch.long)
    board = torch.cat(
        [
            board_payload[pool].index_select(0, board_order).float(),
            feature_payload["board_features"].index_select(0, feat_order).float(),
        ],
        dim=1,
    )
    rows = [rows_by_probe[pid] for pid in common]
    return {
        "probe_id": common,
        "board": board,
        "text_features": feature_payload["text_features"].index_select(0, feat_order).float(),
        "texts": [str(r.get("comment_text") or "") for r in rows],
        "source_family": [str(r.get("source_family") or "unknown") for r in rows],
        "position_id": [str(r.get("position_id")) for r in rows],
        "normalized_text": [normalize_comment(r.get("comment_text")) for r in rows],
    }


class FinetuneTextModel(torch.nn.Module):
    """학습형 MiniLM + (재사용) AlignmentConnector heads."""

    def __init__(self, model_name: str, connector_cfg: ConnectorConfig) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        self.connector = AlignmentConnector(connector_cfg)

    def encode_text_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = torch.nn.functional.normalize(mean_pool(hidden, attention_mask), dim=1)
        return self.connector.encode_text(torch.cat([pooled, text_features], dim=1))


def _encode_all_texts(
    model: FinetuneTextModel,
    tokenizer,
    texts: list[str],
    text_features: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    max_length: int = 128,
) -> torch.Tensor:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoded = tokenizer(
                chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            feats = text_features[start : start + batch_size].to(device)
            outputs.append(model.encode_text_batch(encoded["input_ids"], encoded["attention_mask"], feats).cpu())
    return torch.cat(outputs, dim=0)


def finetune_text_encoder(config: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config.get("seed", 11))
    torch.manual_seed(seed)
    model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    max_length = int(config.get("max_length", 128))
    pool = config.get("pool", "board_pooled")

    train = load_finetune_split(config["train_corpus"], config["train_board"], config["train_features"], pool)
    val = load_finetune_split(config["val_corpus"], config["val_board"], config["val_features"], pool)

    connector_cfg = ConnectorConfig(
        board_dim=train["board"].size(1),
        text_dim=384 + train["text_features"].size(1),
        proj_dim=int(config.get("proj_dim", 128)),
        hidden_dim=int(config.get("hidden_dim", 512)),
        projection=config.get("projection", "mlp"),
        dropout=float(config.get("dropout", 0.1)),
        learnable_temperature=bool(config.get("learnable_temperature", True)),
        init_temperature=float(config.get("init_temperature", 0.07)),
    )
    model = FinetuneTextModel(model_name, connector_cfg).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampler = FamilyBlockedSampler(
        train["source_family"],
        families_per_batch=int(config.get("families_per_batch", 16)),
        samples_per_family=int(config.get("samples_per_family", 4)),
        seed=seed,
    )
    epochs = int(config.get("epochs", 12))
    patience = int(config.get("early_stop_patience", 3))
    encoder_lr = float(config.get("encoder_lr", 2e-5))
    head_lr = float(config.get("head_lr", 1e-3))
    warmup_ratio = float(config.get("warmup_ratio", 0.05))
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.connector.parameters(), "lr": head_lr},
        ],
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def batches_for_epoch(epoch: int) -> list[list[int]]:
        sampler.rng.seed(seed * 1000 + epoch)
        batches = list(iter(sampler))
        if not batches:
            raise ValueError("family_blocked sampler가 batch를 만들지 못했다.")
        return batches

    steps_per_epoch = len(batches_for_epoch(0))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    best_score, best_epoch, best_state, best_metrics = -1.0, -1, None, None
    history: list[dict[str, Any]] = []
    global_step = 0
    stale = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch_indices in batches_for_epoch(epoch):
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * lr_scale(global_step)
            encoded = tokenizer(
                [train["texts"][i] for i in batch_indices],
                padding=True, truncation=True, max_length=max_length, return_tensors="pt",
            ).to(device)
            text_feats = train["text_features"][batch_indices].to(device)
            board = train["board"][batch_indices].to(device)
            zt = model.encode_text_batch(encoded["input_ids"], encoded["attention_mask"], text_feats)
            zb = model.connector.encode_board(board)
            pos_mask, ignore_mask = build_pair_masks(
                [train["position_id"][i] for i in batch_indices],
                [train["normalized_text"][i] for i in batch_indices],
            )
            loss = multi_positive_infonce(
                zb, zt, model.connector.scale(), pos_mask.to(device), ignore_mask.to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
            global_step += 1

        zt_val = _encode_all_texts(model, tokenizer, val["texts"], val["text_features"], device, max_length=max_length)
        with torch.no_grad():
            zb_val = model.connector.encode_board(val["board"].to(device)).cpu()
        metrics = bidirectional_metrics(zb_val, zt_val)
        score = 0.5 * (metrics["text_to_board"]["mrr"] + metrics["board_to_text"]["mrr"])
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss / max(n_batches, 1), "val_selection_score": score, "val": metrics})
        if score > best_score:
            best_score, best_epoch, best_metrics = score, epoch + 1, metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "connector_config": asdict(connector_cfg),
            "model_name": model_name,
            "seed": seed,
            "pool": pool,
            "best_epoch": best_epoch,
            "config": config,
        },
        output_dir / "finetuned_text.pt",
    )
    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_selection_score": best_score,
        "best_val_metrics": best_metrics,
        "history": history,
    }
    (output_dir / "finetune_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_finetuned(config: dict[str, Any]) -> dict[str, Any]:
    """test pool 채점: null control + 비언급 세그먼트 subset 지표 + kill criteria 판정."""
    from transformers import AutoTokenizer

    from modalchess.align.metrics import _ranks_strict
    from modalchess.align.oracle_ceiling import mention_score_matrix, move_attributes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(config["finetuned"], map_location="cpu", weights_only=False)
    connector_cfg = ConnectorConfig(**payload["connector_config"])
    model = FinetuneTextModel(payload["model_name"], connector_cfg)
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(payload["model_name"])

    test = load_finetune_split(
        config["test_corpus"], config["test_board"], config["test_features"],
        pool=payload.get("pool", "board_pooled"),
    )
    zt = _encode_all_texts(model, tokenizer, test["texts"], test["text_features"], device)
    with torch.no_grad():
        zb = model.connector.encode_board(test["board"].to(device)).cpu()

    result = null_control(
        zb, zt, test["source_family"],
        repeats=int(config.get("null_repeats", 50)),
        seed=int(config.get("null_seed", 20260710)),
    )

    # 비언급 세그먼트 subset 지표
    rows_by_probe: dict[str, dict[str, Any]] = {}
    with open(config["test_corpus"], encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows_by_probe[str(row["probe_id"])] = row
    rows = [rows_by_probe[pid] for pid in test["probe_id"]]
    comments = [normalize_comment(str(r.get("comment_text", ""))) for r in rows]
    attrs = [move_attributes(str(r["fen"]), str(r["target_move_uci"])) for r in rows]
    mention = mention_score_matrix(comments, attrs)
    n = len(rows)
    diag = torch.arange(n)
    segment = [i for i in range(n) if mention[i, i] == 0.0]
    seg_idx = torch.tensor(segment, dtype=torch.long)

    def subset_metrics(ranks: torch.Tensor) -> dict[str, float]:
        r = ranks[seg_idx]
        out = {"mrr": float((1.0 / r).mean())}
        out.update({f"r@{k}": float((r <= k).float().mean()) for k in (1, 5, 10, 50)})
        return out

    seg = {
        "n_segment": len(segment),
        "text_to_board": subset_metrics(_ranks_strict(zt @ zb.T, diag).float()),
        "board_to_text": subset_metrics(_ranks_strict(zb @ zt.T, diag).float()),
    }

    wnull = result["within_family_null"]
    seg_mrr = seg["text_to_board"]["mrr"]
    verdict = {
        "segment_t2b_mrr": seg_mrr,
        "segment_gain_vs_hybrid": seg_mrr / BASELINE_SEGMENT_T2B_MRR,
        "passes_segment_min_bar": seg_mrr >= 1.3 * BASELINE_SEGMENT_T2B_MRR,
        "t2b_above_within_family_null": result["real"]["text_to_board"]["mrr"] > wnull["text_to_board_mrr_max"],
        "b2t_above_within_family_null": result["real"]["board_to_text"]["mrr"] > wnull["board_to_text_mrr_max"],
    }
    output = {
        "n_test": n,
        "real": result["real"],
        "global_null": result["global_null"],
        "within_family_null": wnull,
        "segment": seg,
        "verdict": verdict,
    }
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "finetune_eval.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output
