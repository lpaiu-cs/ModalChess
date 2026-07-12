"""P0 (d): 주입 sanity — 캘리브레이션 상수 측정 + 임베딩 주입 안정성 + 채점 경로 검증.

검사: (1) LM 임베딩 per-token RMS 통계 → 캘리브레이션 상수, (2) RMS 정합 무작위 토큰
64개 주입 시 teacher-forced NLL 유한, (3) 주입 상태 greedy 생성 비붕괴, (4) 후보
logprob-ranking 경로 동작. 결과: outputs/phase2/p0_sanity.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modalchess.fusion.prompting import PRE_BOARD, answer_segment, post_board  # noqa: E402

MODEL_DIR = "E:/models/Qwen3-4B-Instruct-2507"


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()

    embed = model.get_input_embeddings()
    weight = embed.weight.detach().float()
    per_token_rms = weight.pow(2).mean(dim=1).sqrt()
    calib_rms = float(per_token_rms.median())
    hidden = weight.size(1)
    print(f"embed: vocab={weight.size(0)} hidden={hidden} "
          f"rms median={calib_rms:.6f} mean={per_token_rms.mean():.6f} std={per_token_rms.std():.6f}")

    def ids(text: str) -> torch.Tensor:
        return torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]], device=device)

    def embeds(text: str) -> torch.Tensor:
        return embed(ids(text))

    question = "Is the side to move currently in check?"
    pre = embeds(PRE_BOARD)
    post = embeds(post_board(question))
    board_inj = (torch.randn(1, 64, hidden, generator=torch.Generator().manual_seed(11))
                 .to(device, torch.bfloat16) * calib_rms)

    # (2) teacher-forced NLL
    answer_ids = ids(answer_segment("no"))
    answer_emb = embed(answer_ids)
    inputs = torch.cat([pre, board_inj, post, answer_emb], dim=1)
    labels = torch.full((1, inputs.size(1)), -100, device=device, dtype=torch.long)
    labels[0, -answer_ids.size(1):] = answer_ids[0]
    with torch.no_grad():
        out = model(inputs_embeds=inputs, labels=labels)
    nll = float(out.loss)
    print(f"teacher-forced answer NLL (random board tokens): {nll:.4f} finite={torch.isfinite(out.loss).item()}")

    # (3) greedy 생성 비붕괴
    prompt = torch.cat([pre, board_inj, post], dim=1)
    with torch.no_grad():
        gen = model.generate(
            inputs_embeds=prompt,
            attention_mask=torch.ones(prompt.shape[:2], device=device, dtype=torch.long),
            max_new_tokens=24, do_sample=False,
        )
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    degenerate = len(set(gen[0].tolist())) <= 2
    print(f"generation: {text!r} degenerate={degenerate}")

    # (4) 후보 logprob-ranking 경로
    def candidate_logprob(candidate: str) -> float:
        cand_ids = ids(candidate + "<|im_end|>")
        cand_emb = embed(cand_ids)
        full = torch.cat([prompt, cand_emb], dim=1)
        with torch.no_grad():
            logits = model(inputs_embeds=full).logits
        logprobs = torch.log_softmax(logits[0, prompt.size(1) - 1:-1].float(), dim=-1)
        return float(logprobs.gather(1, cand_ids[0].unsqueeze(1)).sum())

    scores = {c: candidate_logprob(c) for c in ("yes", "no")}
    print(f"logprob ranking smoke: {scores}")

    result = {
        "model_dir": MODEL_DIR,
        "hidden": hidden,
        "calibration_rms": calib_rms,
        "rms_mean": float(per_token_rms.mean()),
        "rms_std": float(per_token_rms.std()),
        "teacher_forced_nll_random_board": nll,
        "nll_finite": bool(torch.isfinite(out.loss)),
        "generation_sample": text,
        "generation_degenerate": degenerate,
        "logprob_smoke": scores,
    }
    out_path = ROOT / "outputs/phase2/p0_sanity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = result["nll_finite"] and not degenerate
    print(f"P0_SANITY_{'PASS' if passed else 'FAIL'}")
    if not passed:
        # CI/배치 게이트가 exit code에 의존 — 실패한 캘리브레이션으로 P1 진입 차단
        raise SystemExit(1)


if __name__ == "__main__":
    main()
