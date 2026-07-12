"""P1 학습·평가 러너 (docs/phase2_plan.md §3, §6).

시퀀스 조립: [PRE][주입 64(해당 arm)][FEN 텍스트(해당 arm)][POST+question][answer]
- 학습: answer 토큰만 CE. LM·백본 동결, arm 학습 파라미터(projection/soft)만 갱신.
- 평가: 후보별 로그확률 합 argmax (형식 게이밍 제거). shuffled-board null 포함.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Any

import torch

from modalchess.fusion.fusion_arms import (
    ARM_KINDS,
    FrozenBoardBackbone,
    FusionArm,
    fen_to_planes_meta,
)
from modalchess.fusion.prompting import PRE_BOARD, answer_segment, fen_board_text, post_board

N_INJECT = 64


def load_qa_items(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


class SequenceAssembler:
    """토큰 시퀀스 조립 — 전 arm 공통 골격 (주입 슬롯 위치는 PRE 직후로 고정)."""

    def __init__(self, tokenizer, inject: bool, fen_text: bool) -> None:
        self.tokenizer = tokenizer
        self.inject = inject
        self.fen_text = fen_text
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        self.pre_ids: list[int] = tokenizer(PRE_BOARD, add_special_tokens=False)["input_ids"]

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def build(
        self,
        items: list[dict[str, Any]],
        answers: list[str],
        device: torch.device,
        fens_override: list[str] | None = None,
    ) -> dict[str, Any]:
        """items×answers → 배치 텐서. fens_override는 shuffled-null용 보드 소스 치환."""
        rows: list[dict[str, Any]] = []
        for item, answer in zip(items, answers):
            fen = (fens_override[len(rows)] if fens_override is not None else item["fen"])
            mid_ids = self._ids(fen_board_text(fen)) if self.fen_text else []
            post_ids = self._ids(post_board(item["question"]))
            ans_ids = self._ids(answer_segment(answer))
            inj = [self.pad_id] * N_INJECT if self.inject else []
            ids = self.pre_ids + inj + mid_ids + post_ids + ans_ids
            ans_start = len(ids) - len(ans_ids)
            rows.append({"ids": ids, "ans_start": ans_start, "ans_len": len(ans_ids)})
        max_len = max(len(r["ids"]) for r in rows)
        input_ids = torch.full((len(rows), max_len), self.pad_id, dtype=torch.long)
        attention = torch.zeros((len(rows), max_len), dtype=torch.long)
        labels = torch.full((len(rows), max_len), -100, dtype=torch.long)
        for i, row in enumerate(rows):
            n = len(row["ids"])
            input_ids[i, :n] = torch.tensor(row["ids"], dtype=torch.long)
            attention[i, :n] = 1
            s, l = row["ans_start"], row["ans_len"]
            labels[i, s : s + l] = input_ids[i, s : s + l]
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention.to(device),
            "labels": labels.to(device),
            "inject_slice": (len(self.pre_ids), N_INJECT) if self.inject else None,
            "ans_spans": [(r["ans_start"], r["ans_len"]) for r in rows],
            "n": len(rows),
        }


def board_batch(items: list[dict[str, Any]], history_length: int,
                fens_override: list[str] | None = None) -> dict[str, torch.Tensor]:
    planes, metas = [], []
    for i, item in enumerate(items):
        fen = fens_override[i] if fens_override is not None else item["fen"]
        p, m = fen_to_planes_meta(fen, history_length)
        planes.append(p)
        metas.append(m)
    return {"planes": torch.stack(planes), "meta": torch.stack(metas)}


def _forward_embeds(model, embed_layer, assembled, arm: FusionArm,
                    arm_inputs: dict[str, Any], device: torch.device) -> torch.Tensor:
    embeds = embed_layer(assembled["input_ids"])
    if assembled["inject_slice"] is not None:
        start, length = assembled["inject_slice"]
        injected = arm.injected({**arm_inputs, "n": assembled["n"]}, device)
        embeds = embeds.clone()
        embeds[:, start : start + length] = injected.to(embeds.dtype)
    return embeds


def train_arm(config: dict[str, Any], arm: FusionArm, model, tokenizer,
              device: torch.device) -> dict[str, Any]:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)  # 체크포인트 저장 대상 디렉터리
    assembler = SequenceAssembler(tokenizer, inject=arm.kind != "fen_zs",
                                  fen_text=arm.uses_fen_text)
    train_items = load_qa_items(Path(config["qa_dir"]) / "qa_train.jsonl",
                                config.get("limit_train"))
    val_items = load_qa_items(Path(config["qa_dir"]) / "qa_val.jsonl",
                              config.get("limit_val"))
    val_subset = val_items[: int(config.get("val_subset", 2000))]

    embed_layer = model.get_input_embeddings()
    params = arm.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=float(config.get("lr", 1e-3)),
                                  weight_decay=float(config.get("weight_decay", 0.01)))
    batch_size = int(config.get("batch_size", 16))
    # micro_batch_size < batch_size면 gradient accumulation으로 유효 배치를 유지한다
    # (FEN 텍스트 arm은 시퀀스가 길어 어텐션 메모리가 O(len^2)로 커짐 — micro를 줄여 대응).
    micro_batch = int(config.get("micro_batch_size", batch_size))
    micro_batch = max(1, min(micro_batch, batch_size))
    epochs = int(config.get("epochs", 2))
    steps_per_epoch = math.ceil(len(train_items) / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.05)))
    base_lr = float(config.get("lr", 1e-3))
    history_length = arm.backbone.history_length if arm.backbone is not None else 1

    def lr_at(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    best = {"score": -1.0, "state": None, "epoch": -1}
    history: list[dict[str, Any]] = []
    global_step = 0
    start_epoch = 0

    # epoch별 재개: train_checkpoint.pt가 있으면 마지막 완료 epoch 다음부터 이어간다.
    ckpt_path = Path(config["output_dir"]) / "train_checkpoint.pt"
    if config.get("resume", True) and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt.get("seed") == seed and ckpt.get("arm") == arm.kind:
            arm.load_state_dict(ckpt["arm_state"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer_state"])
            best = ckpt["best"]
            history = ckpt["history"]
            global_step = ckpt["global_step"]
            start_epoch = ckpt["epoch"] + 1
            print(f"[resume] {arm.kind}: epoch {ckpt['epoch']} 완료 지점에서 재개 "
                  f"(best_val={best['score']:.4f}, global_step={global_step})", flush=True)

    def save_checkpoint(epoch: int) -> None:
        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save(
            {"arm": arm.kind, "seed": seed, "epoch": epoch, "global_step": global_step,
             "arm_state": {k: v.detach().cpu().clone()
                           for k, v in arm.state_dict().items()
                           if not k.startswith("backbone.")},
             "optimizer_state": optimizer.state_dict(),
             "best": best, "history": history},
            tmp,
        )
        tmp.replace(ckpt_path)  # 원자적 교체 — 저장 도중 kill돼도 이전 체크포인트 보존

    for epoch in range(start_epoch, epochs):
        rng = random.Random(seed * 1000 + epoch)
        order = list(range(len(train_items)))
        rng.shuffle(order)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch_items = [train_items[i] for i in order[start : start + batch_size]]
            for group in optimizer.param_groups:
                group["lr"] = lr_at(global_step)
            optimizer.zero_grad(set_to_none=True)
            # 유효 배치를 micro-batch로 쪼개 누적(grad accumulation). loss를 micro 수로
            # 나눠 backward → 그래디언트 합은 full-batch 평균과 동등, 메모리는 micro 크기.
            micro_chunks = list(range(0, len(batch_items), micro_batch))
            batch_loss = 0.0
            for ms in micro_chunks:
                micro_items = batch_items[ms : ms + micro_batch]
                answers = [it["answer"] for it in micro_items]
                assembled = assembler.build(micro_items, answers, device)
                arm_inputs = (board_batch(micro_items, history_length)
                              if arm.uses_board_planes else {})
                embeds = _forward_embeds(model, embed_layer, assembled, arm, arm_inputs, device)
                out = model(inputs_embeds=embeds,
                            attention_mask=assembled["attention_mask"],
                            labels=assembled["labels"])
                (out.loss / len(micro_chunks)).backward()
                batch_loss += float(out.loss.detach())
            torch.nn.utils.clip_grad_norm_(params, float(config.get("grad_clip", 1.0)))
            optimizer.step()
            epoch_loss += batch_loss / len(micro_chunks)
            n_batches += 1
            global_step += 1
            if global_step % 200 == 0:
                print(f"  step {global_step}/{total_steps} loss={epoch_loss / n_batches:.4f}",
                      flush=True)
        val_metrics = evaluate_items(config, arm, model, tokenizer, device, val_subset)
        score = val_metrics["overall"]["accuracy"]
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss / max(n_batches, 1),
                        "val_accuracy": score})
        print(f"epoch {epoch + 1}: loss={epoch_loss / max(n_batches, 1):.4f} "
              f"val_acc={score:.4f}", flush=True)
        if score > best["score"]:
            best = {"score": score,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in arm.state_dict().items()
                              if not k.startswith("backbone.")},
                    "epoch": epoch + 1}
        save_checkpoint(epoch)  # epoch 경계마다 저장 → 중단 시 최대 1 epoch만 손실
    if best["state"] is not None:
        arm.load_state_dict(best["state"], strict=False)
    return {"best_val_accuracy": best["score"], "best_epoch": best["epoch"],
            "history": history}


@torch.no_grad()
def evaluate_items(config: dict[str, Any], arm: FusionArm, model, tokenizer,
                   device: torch.device, items: list[dict[str, Any]],
                   shuffle_board_seed: int | None = None) -> dict[str, Any]:
    """후보 logprob 채점. shuffle_board_seed 지정 시 보드 채널만 파생 순열로 치환(null).

    hybrid(FEN 텍스트+보드 토큰)에서는 **보드 토큰(시각 채널)만 셔플하고 FEN 텍스트는 참값
    유지** — null이 시각 주입 채널을 격리하도록. 순수 FEN arm은 FEN 텍스트가 유일한 보드
    채널이므로 텍스트를 셔플한다.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()  # 학습 캐시 반환 후 eval — 단편화로 인한 shared 스필 방지
    assembler = SequenceAssembler(tokenizer, inject=arm.kind != "fen_zs",
                                  fen_text=arm.uses_fen_text)
    embed_layer = model.get_input_embeddings()
    history_length = arm.backbone.history_length if arm.backbone is not None else 1
    eval_bs = int(config.get("eval_batch_size", 48))

    true_fens = [it["fen"] for it in items]
    planes_fen = true_fens  # board_batch(주입 토큰) 소스
    text_fen = true_fens    # assembler(FEN 텍스트) 소스
    if shuffle_board_seed is not None:
        rng = random.Random(shuffle_board_seed)
        perm = list(range(len(items)))
        rng.shuffle(perm)
        perm = perm[1:] + perm[:1]  # 고정점 제거(derangement 근사)
        shuffled_fens = [items[j]["fen"] for j in perm]
        planes_fen = shuffled_fens
        # FEN 텍스트가 유일한 보드 채널인 arm만 텍스트도 셔플. hybrid는 텍스트 참값 유지.
        text_fen = shuffled_fens if (arm.uses_fen_text and not arm.uses_board_planes) else true_fens

    rows: list[tuple[int, str]] = []
    for idx, item in enumerate(items):
        for cand in item["candidates"]:
            rows.append((idx, cand))

    scores: dict[int, list[tuple[str, float]]] = {i: [] for i in range(len(items))}
    for start in range(0, len(rows), eval_bs):
        chunk = rows[start : start + eval_bs]
        chunk_items = [items[i] for i, _ in chunk]
        chunk_answers = [c for _, c in chunk]
        chunk_text_fens = [text_fen[i] for i, _ in chunk]
        chunk_plane_fens = [planes_fen[i] for i, _ in chunk]
        assembled = assembler.build(chunk_items, chunk_answers, device,
                                    fens_override=chunk_text_fens if arm.uses_fen_text else None)
        arm_inputs = (board_batch(chunk_items, history_length, fens_override=chunk_plane_fens)
                      if arm.uses_board_planes else {})
        embeds = _forward_embeds(model, embed_layer, assembled, arm, arm_inputs, device)
        logits = model(inputs_embeds=embeds,
                       attention_mask=assembled["attention_mask"]).logits
        # 메모리 저부하: 전체 [B,seq,vocab]를 float로 캐스트하지 않고 답 토큰 위치
        # 슬라이스에만 log_softmax를 건다. log_softmax는 vocab 차원 전체에 대해
        # 계산되므로 결과는 전체 캐스트와 수치적으로 동일하다(사전 등록 채점 불변).
        for r, ((item_idx, cand), (ans_start, ans_len)) in enumerate(
            zip(chunk, assembled["ans_spans"])
        ):
            token_ids = assembled["input_ids"][r, ans_start : ans_start + ans_len]
            span = logits[r, ans_start - 1 : ans_start + ans_len - 1].float()
            lp = torch.log_softmax(span, dim=-1)
            total = float(lp.gather(1, token_ids.unsqueeze(1)).sum())
            scores[item_idx].append((cand, total))
        del logits, embeds

    correct_flags: list[bool] = []
    for idx, item in enumerate(items):
        ranked = sorted(scores[idx], key=lambda x: -x[1])
        correct_flags.append(ranked[0][0] == item["answer"])
    return aggregate_metrics(items, correct_flags)


def aggregate_metrics(items: list[dict[str, Any]], correct: list[bool]) -> dict[str, Any]:
    def acc(indices: list[int]) -> dict[str, Any]:
        if not indices:
            return {"n": 0, "accuracy": float("nan")}
        return {"n": len(indices),
                "accuracy": sum(correct[i] for i in indices) / len(indices)}

    by_task: dict[str, list[int]] = {}
    by_tier: dict[str, list[int]] = {}
    held_out: list[int] = []
    for i, item in enumerate(items):
        by_task.setdefault(item["task"], []).append(i)
        by_tier.setdefault(item["tier"], []).append(i)
        if item["template_id"] == 3:
            held_out.append(i)
    return {
        "overall": acc(list(range(len(items)))),
        "tiers": {t: acc(ix) for t, ix in sorted(by_tier.items())},
        "tasks": {t: acc(ix) for t, ix in sorted(by_task.items())},
        "held_out_template": acc(held_out),
    }


def run_arm(config: dict[str, Any], arm_kind: str) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if arm_kind not in ARM_KINDS:
        raise ValueError(f"unknown arm: {arm_kind}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config["seed"])
    torch.manual_seed(seed)

    # arm-level 재개: 이미 완료된 동일 arm·seed(eval.json + test)이면 재실행하지 않고 건너뛴다.
    # arm_kind/seed 일치를 반드시 확인 — 공유 output_dir에서 다른 arm 결과를 재사용하는 것을 차단.
    out_dir = Path(config["output_dir"])
    done_path = out_dir / "eval.json"
    if config.get("resume", True) and done_path.exists():
        try:
            prev = json.loads(done_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = None
        if (prev and prev.get("arm") == arm_kind and prev.get("seed") == seed
                and prev.get("test", {}).get("overall", {}).get("accuracy") is not None):
            print(f"ARM_SKIP {arm_kind} seed={seed} "
                  f"test_acc={prev['test']['overall']['accuracy']:.4f} (already done)", flush=True)
            return prev

    calib = json.loads(Path(config["calib_json"]).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(config["model_dir"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_dir"], dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.eval()

    backbone = None
    if arm_kind in ("board", "hybrid"):
        backbone = FrozenBoardBackbone(config["encoder_checkpoint"]).to(device)
    arm = FusionArm(
        kind=arm_kind, d_lm=int(calib["hidden"]), calib_rms=float(calib["calibration_rms"]),
        proj_hidden=int(config.get("proj_hidden", 5120)), backbone=backbone,
    ).to(device)

    n_trainable = sum(p.numel() for p in arm.trainable_parameters())
    print(f"arm={arm_kind} seed={seed} trainable_params={n_trainable}", flush=True)

    summary: dict[str, Any] = {"arm": arm_kind, "seed": seed,
                               "n_trainable_params": n_trainable}
    if arm_kind != "fen_zs":
        summary["train"] = train_arm(config, arm, model, tokenizer, device)

    test_items = load_qa_items(Path(config["qa_dir"]) / "qa_test.jsonl",
                               config.get("limit_test"))
    summary["test"] = evaluate_items(config, arm, model, tokenizer, device, test_items)
    if arm_kind != "blind":
        summary["test_shuffled_null"] = evaluate_items(
            config, arm, model, tokenizer, device, test_items,
            shuffle_board_seed=int(config.get("null_seed", 20260710)),
        )

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if arm_kind != "fen_zs":
        torch.save(
            {"state": {k: v for k, v in arm.state_dict().items()
                       if not k.startswith("backbone.")},
             "config": config, "arm": arm_kind, "seed": seed},
            out_dir / "arm_state.pt",
        )
    (out_dir / "eval.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ckpt = out_dir / "train_checkpoint.pt"
    if ckpt.exists():
        ckpt.unlink()  # arm 완료 — 재개 체크포인트 불필요
    print(f"ARM_DONE {arm_kind} seed={seed} "
          f"test_acc={summary['test']['overall']['accuracy']:.4f}", flush=True)
    return summary
