"""P1 하네스 순수 로직 테스트: 좌표 정렬, 시퀀스/라벨 스팬, 주입 스케일."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from modalchess.fusion.fusion_arms import FusionArm, ProjectionMLP, fen_to_planes_meta, raw_square_features

MODEL_DIR = Path("E:/models/Qwen3-4B-Instruct-2507")
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_raw_square_features_ordering() -> None:
    planes, meta = fen_to_planes_meta(START, history_length=1)
    raw = raw_square_features(planes.unsqueeze(0))
    assert raw.shape == (1, 64, 18)
    # a1(토큰 0) = 텐서 좌표 row7,col0 / h8(토큰 63) = row0,col7 (AGENTS.md 좌표 규약)
    assert torch.equal(raw[0, 0], planes[0, :, 7, 0])
    assert torch.equal(raw[0, 63], planes[0, :, 0, 7])
    assert meta.shape == (3,)


def test_projection_rms_calibration() -> None:
    proj = ProjectionMLP(d_in=18, d_lm=64, hidden=32, calib_rms=0.022)
    out = proj(torch.randn(2, 64, 18))
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 0.022), atol=1e-4)


def test_blind_arm_injection_shape() -> None:
    arm = FusionArm(kind="blind", d_lm=32, calib_rms=0.02, proj_hidden=16)
    out = arm.injected({"n": 3}, torch.device("cpu"))
    assert out.shape == (3, 64, 32)
    assert len(arm.trainable_parameters()) == 1


def test_resolve_arm_dir_isolates_and_is_idempotent() -> None:
    from pathlib import Path

    from modalchess.fusion.fusion_run import resolve_arm_dir

    # 공유 base → arm/seed 하위 디렉터리로 격리 (다른 arm은 다른 경로)
    assert resolve_arm_dir("out/p1", "board", 11) == Path("out/p1/board_seed11")
    assert resolve_arm_dir("out/p1", "fen_soft", 11) == Path("out/p1/fen_soft_seed11")
    # 이미 arm-specific 경로면 중복 append 안 함(러너가 arm 디렉터리를 직접 넘겨도 안전)
    assert resolve_arm_dir("out/p1/board_seed11", "board", 11) == Path("out/p1/board_seed11")


def test_hybrid_arm_uses_both_channels() -> None:
    # hybrid는 board 토큰 주입 + FEN 텍스트 병행 (백본 없이 속성만 검증)
    class _StubBackbone:
        d_model = 8
        history_length = 1

    arm = FusionArm(kind="hybrid", d_lm=32, calib_rms=0.02, proj_hidden=16,
                    backbone=_StubBackbone())
    assert arm.uses_fen_text and arm.uses_board_planes
    # 학습 파라미터는 projection(board 토큰용)뿐 — soft 토큰 없음
    assert not hasattr(arm, "soft_tokens")
    assert len(arm.trainable_parameters()) > 0


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="local LM not present")
def test_sequence_assembler_spans() -> None:
    from transformers import AutoTokenizer

    from modalchess.fusion.fusion_run import N_INJECT, SequenceAssembler
    from modalchess.fusion.prompting import answer_segment

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    items = [
        {"fen": START, "question": "Whose turn is it to move?"},
        {"fen": START, "question": "Is the side to move currently in check?"},
    ]
    assembler = SequenceAssembler(tokenizer, inject=True, fen_text=False)
    batch = assembler.build(items, ["white", "no"], torch.device("cpu"))

    start, length = batch["inject_slice"]
    assert length == N_INJECT and start == len(assembler.pre_ids)
    for row, (item, answer) in enumerate(zip(items, ["white", "no"])):
        span_start, span_len = batch["ans_spans"][row]
        label_positions = (batch["labels"][row] != -100).nonzero().flatten().tolist()
        assert label_positions == list(range(span_start, span_start + span_len))
        decoded = tokenizer.decode(batch["input_ids"][row, span_start : span_start + span_len])
        assert decoded == answer_segment(answer)
    # FEN 텍스트 arm은 주입 슬롯 뒤에 FEN이 들어가 시퀀스가 더 길다
    fen_assembler = SequenceAssembler(tokenizer, inject=True, fen_text=True)
    fen_batch = fen_assembler.build(items[:1], ["white"], torch.device("cpu"))
    assert fen_batch["input_ids"].size(1) > batch["input_ids"].size(1)
