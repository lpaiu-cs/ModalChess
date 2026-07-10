# Phase 2 로그 — 시각 모달리티 (사전 등록: [phase2_plan.md](phase2_plan.md))

## 2026-07-10 — P0: 데이터·주입 인프라 — 통과

- **QA 코퍼스** (`scripts/build_qa_corpus.py`, seed 20260710): train 116,997 / val 5,851 /
  test 11,693 items (9과제, T1×5+T2×4).
  - (a) **독립 검증 불일치 0 / 134,541** — 생성기·검증기가 서로 다른 python-chess API
    경로로 정답 계산(piece_at↔piece_map, is_attacked_by↔attackers, is_pinned↔pin 마스크,
    캐슬링 rights API↔FEN 필드 파싱 등).
  - (b) 클래스 쿼터 **완전 충전(균형 편차 0)**, 거짓 전제(no-such-piece) 1/3 포함.
  - (c) split 위생: D1의 game overlap **0** (train∩val, train∩test, val∩test 모두),
    포지션 키 크로스-split dedup(test>val>train). QA-test는 supervised_test에서만 —
    **인코더 미노출 포지션**. test에 held-out 템플릿 항목 2,957개.
- **주입 sanity** (`scripts/p0_injection_sanity.py`): Qwen3-4B-Instruct-2507
  (revision `cdbee75f` 핀, `E:/models/`), 임베딩 RMS 캘리브레이션 상수 **0.02202**.
  RMS 정합 무작위 토큰 64개 주입 → teacher-forced NLL 유한(8.65), greedy 생성 비붕괴
  (모델이 "보드가 깨진 문자열로 보인다"고 유창하게 응답 — 주입 경로 안정 확인),
  후보 logprob-ranking 경로 동작.
- 개정 1 적용(데이터 열람 전): piece_count 후보 3-way.
- 판정: **P0 통과.** (이 게이트는 모델 능력에 대해 아무것도 증명하지 않는다.)

## P1 하네스 구현 메모

- `fusion_arms.py`: FrozenBoardBackbone(G3 seed11 best_grounding, meta+encoder 경로 재현,
  `tokens`[64,384]), ProjectionMLP(LN→5120→GELU→2560→RMSScale), soft 토큰 arm.
- `fusion_run.py`: 시퀀스 [PRE][주입64][FEN텍스트?][question][answer], answer-only CE,
  후보 logprob 합 argmax 채점, shuffled-board null(파생 순열), 과제/티어/held-out 템플릿
  슬라이스 집계. 전 arm 동일 골격·동일 lr(스윕 금지).
- smoke(극소 한도, board arm): 학습·채점·null·저장 전 경로 정상. projection 15.08M 파라미터.
