"""Connector 학습 루프: precompute된 board/text 위에서 multi-positive InfoNCE.

early stop 기준은 val `mean(strict_t2b_mrr, strict_b2t_mrr)` — t2b 편향을 피한다.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import torch

from modalchess.align.connector import AlignmentConnector, ConnectorConfig, multi_positive_infonce
from modalchess.align.dataset import FamilyBlockedSampler, build_batch_masks, load_aligned_pairs
from modalchess.align.metrics import bidirectional_metrics
from modalchess.align.text_embed import normalize_comment  # noqa: F401  (계약 문서화용)


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _val_selection_score(pairs, model, device) -> tuple[float, dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        zb = model.encode_board(pairs.board.to(device)).cpu()
        zt = model.encode_text(pairs.text.to(device)).cpu()
    metrics = bidirectional_metrics(zb, zt)
    score = 0.5 * (metrics["text_to_board"]["mrr"] + metrics["board_to_text"]["mrr"])
    return score, metrics


def train_connector(config: dict[str, Any]) -> dict[str, Any]:
    device = _resolve_device()
    seed = int(config.get("seed", 11))
    torch.manual_seed(seed)

    pool = config.get("pool", "board_pooled")
    train_pairs = load_aligned_pairs(config["train_board"], config["train_text"], pool=pool)
    val_pairs = load_aligned_pairs(config["val_board"], config["val_text"], pool=pool)

    model_cfg = ConnectorConfig(
        board_dim=train_pairs.board.size(1),
        text_dim=train_pairs.text.size(1),
        proj_dim=int(config.get("proj_dim", 128)),
        hidden_dim=int(config.get("hidden_dim", 512)),
        projection=config.get("projection", "mlp"),
        dropout=float(config.get("dropout", 0.1)),
        learnable_temperature=bool(config.get("learnable_temperature", True)),
        init_temperature=float(config.get("init_temperature", 0.07)),
    )
    model = AlignmentConnector(model_cfg).to(device)

    balance = config.get("balance", "family_blocked")
    if balance == "family_blocked":
        sampler = FamilyBlockedSampler(
            train_pairs.source_family,
            families_per_batch=int(config.get("families_per_batch", 16)),
            samples_per_family=int(config.get("samples_per_family", 4)),
            seed=seed,
        )
        use_multi_positive = bool(config.get("multi_positive", True))
    else:
        sampler = None
        use_multi_positive = bool(config.get("multi_positive", True))

    epochs = int(config.get("epochs", 40))
    lr = float(config.get("learning_rate", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.01))
    warmup_ratio = float(config.get("warmup_ratio", 0.05))
    patience = int(config.get("early_stop_patience", 6))
    batch_size = int(config.get("batch_size", 512))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def batches_for_epoch(epoch: int) -> list[list[int]]:
        if sampler is not None:
            sampler.rng.seed(seed * 1000 + epoch)
            batches = list(iter(sampler))
            if not batches:
                raise ValueError(
                    "family_blocked sampler가 batch를 만들지 못했다. "
                    "source_family 분포와 families_per_batch/samples_per_family 설정을 확인하라."
                )
            return batches
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(len(train_pairs), generator=generator).tolist()
        return [order[i : i + batch_size] for i in range(0, len(order), batch_size) if len(order[i : i + batch_size]) > 1]

    steps_per_epoch = max(1, len(batches_for_epoch(0)))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_at(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    best_score = -1.0
    best_epoch = -1
    best_state = None
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    global_step = 0
    epochs_without_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_indices in batches_for_epoch(epoch):
            for group in optimizer.param_groups:
                group["lr"] = lr * lr_at(global_step)
            board = train_pairs.board[batch_indices].to(device)
            text = train_pairs.text[batch_indices].to(device)
            if use_multi_positive:
                pos_mask, ignore_mask = build_batch_masks(train_pairs, batch_indices)
            else:
                size = len(batch_indices)
                pos_mask = torch.eye(size, dtype=torch.bool)
                ignore_mask = torch.zeros((size, size), dtype=torch.bool)
            pos_mask = pos_mask.to(device)
            ignore_mask = ignore_mask.to(device)
            zb, zt = model(board, text)
            loss = multi_positive_infonce(zb, zt, model.scale(), pos_mask, ignore_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
            global_step += 1

        score, val_metrics = _val_selection_score(val_pairs, model, device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(n_batches, 1),
            "val_selection_score": score,
            "val": val_metrics,
            "lr": lr * lr_at(global_step - 1),
        })
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_metrics = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({
        "connector_state_dict": model.state_dict(),
        "model_config": asdict(model_cfg),
        "seed": seed,
        "pool": pool,
        "best_epoch": best_epoch,
        "config": config,
    }, output_dir / "connector.pt")
    summary = {
        "seed": seed,
        "pool": pool,
        "projection": model_cfg.projection,
        "proj_dim": model_cfg.proj_dim,
        "balance": balance,
        "multi_positive": use_multi_positive,
        "best_epoch": best_epoch,
        "best_val_selection_score": best_score,
        "best_val_metrics": best_metrics,
        "history": history,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
