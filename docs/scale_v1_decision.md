# scale_v1 결정 기록 (ADR)

- 상태: active
- 일자: 2026-07-10
- 브랜치: `scale_v1` (origin)
- 상세 로그: [scaleup_log.md](scaleup_log.md) (시간순), 본 문서는 결론 고정용

## 맥락

week18까지 언어 신호(board↔comment retrieval)가 잡음 수준이었고
(`STILL_EVAL_ONLY_BUT_STABLE`), 15주간 데이터·평가만 정제했다. 근본 가설:

> **H1. 신호 부재의 주원인은 corpus 품질이 아니라 backbone 용량·학습량 부족이다.**

기존 backbone(G1/G3)은 531k 파라미터 / 96k positions / 2 epochs로, 모든 seed에서
best epoch == last epoch(수렴 전 중단)였다. scale_v1은 평가 인프라를 동결한 채
데이터·모델·학습량을 올려 이 가설을 검증했다.

## 무엇을 했나

1. **인프라** (커밋 3da396b, 1fe7fec, b6a2c67): lazy JSONL dataset, DataLoader 병렬화,
   listwise 손실 벡터화 + fp32 고정(bf16 발산 근본 수정), warmup_cosine + early stop,
   PGN `--min-rating`. Windows commit-limit(error 1455) 대응(persistent_workers=False 등).
2. **데이터**: 2015-01 Lichess 덤프에서 D1(무필터 2.0M positions) 채택. Elo A/B로 확정.
3. **스케일 사다리**: Tier S(d256/6L, 5.1M) → Tier M(d384/8L, 15M). M이 S 대비 NLL 개선
   0.93%(<2%) → 중단 규칙 발동, **모델 축 종료, 승자 = Tier M**.
4. **공식 3-seed**: Tier M × {G1, G3} × seeds {11,17,23}, fp32.
5. **Gate 2 (frozen probe)**: language_probe_v2(MATE/puzzle) + week17 동결 comment regime,
   permutation null control.
6. **옵션 B (텍스트 축)**: tf-idf → all-MiniLM-L6-v2 문장 인코더 2×2 요인 실험.

## 결과 (핵심 수치)

### Gate 1 — 백본 품질: 통과 (3-seed)
| | top-1 | NLL | legality AP |
|---|---|---|---|
| G1 | 0.4746 ± 0.0013 | 1.670 | 0.013 |
| G3 | 0.4747 ± 0.0003 | 1.672 | 0.991 |
| 목표 | ≥0.42 | ≤2.1 | ≥0.60 |

기존 백본 대비: top-1 0.318→0.475, NLL 2.41→1.67, legality AP 0.415→0.991.
정책 손실은 legality 감독 유무에 무관(G1≈G3).

### Gate 2 — 표현 전이: 부분 통과
- language_probe_v2: 새 백본이 기존 대비 16 config 전부 2~3.7× 개선. MATE text→board
  strict R@1 35× 무작위, puzzle 44× 무작위.
- comment regime(정렬 3000행, permutation null): text→board 12 config 전부 real > null max.
- **board→text는 두 regime 모두 near-chance** — 신호는 진짜지만 절대값 약함, 일방향적.

### 옵션 B — 텍스트 축 2×2 요인 (comment text→board strict MRR)
| | tf-idf | 문장 인코더 | 텍스트축 |
|---|---|---|---|
| 기존 백본 | 0.00559 | 0.00692 | 1.24× |
| 새 백본 | 0.00743 | 0.01084 | 1.46× |
| 백본 | 1.33× | 1.57× | |

기준선(0.00559)→ new+sentence 0.01084 = **1.94× 복합** (두 축 곱셈적). new+sentence는
12 config 전부 null max 상회(null mean 대비 3.82×).

## 결정

1. **H1 확증**: 백본은 실재 병목이었다. 스케일업이 신호를 permutation null 위로 밀어올렸다.
2. **백본 단독 병목 아님**: 텍스트 축(tf-idf BoW)이 대등한 병목이며 백본과 복합한다.
   두 축을 함께 풀면 신호가 거의 2배.
3. **Gate 3 = GO (조건부)**: frozen-probe 0.0108은 학습형 connector의 하한. 양축이 독립
   기여·복합하므로, board tap + 텍스트 인코더를 공동 최적화하는 small contrastive
   connector 학습에 착수한다.
4. **금지**: full LLM fusion, rationale generation training, RL은 여전히 out of scope(stub 유지).
   다음은 오직 작은 frozen-기반 connector다.

## 정직 캐비엇 (남은 리스크)

- 절대 성능: 0.0108 MRR은 3000 후보 중 정답 ~90등 — usable top-k retrieval 미달.
- board→text 방향은 near-chance로 남아 board↔comment 정렬에 근본 상한 가능성.
- 따라서 connector의 목표는 "논문용 멀티모달 모델"이 아니라 **"정렬이 학습으로 top-k
  근처까지 오르는지 검증하는 최소 connector"**다.

## 다음 단계 (connector 설계 계약)

1. frozen Tier M board encoder + frozen 문장 인코더 위 small projection connector.
2. 손실: symmetric InfoNCE. batch: source-family balanced.
3. validation: strict MRR/R@K + permutation null + source-holdout + large-pool shared queries.
4. **조기 stop gate**: new+sentence frozen-probe(0.0108)보다 명확히 못 오르거나 board→text가
   계속 null 근처면, connector 구조보다 data alignment 한계를 먼저 의심한다.
5. backbone 후보: **G3 기본 / G1 control**. 주의 — 언어 신호에서 G1≈G3(중립)이며, G3 선택은
   언어 근거가 아니라 downstream substrate(legality AP 0.99, 정책 손실 0) 베팅이다.

## Gate 4 (connector_v1, 2026-07-10) — 부분 통과

frozen Tier M board + frozen MiniLM 위 small contrastive connector 구현·실행(3-seed, G1/G3,
mlp+linear). comment regime 3000행 test:
- **within-family permutation null을 9/9 런 양방향 통과** → 학습형 board↔comment 정렬이
  family/style shortcut이 아니라 real임을 재현적으로 확증.
- b2t(기존 near-chance)를 frozen-probe 대비 3.3×로 rescue, mean 1.73×로 min-bar 통과.
- 그러나 절대 retrieval R@50 ~8%로 usable top-k 미달, t2b는 frozen-probe 소폭 상회.
- 판정: 최소 connector 목적("정렬이 학습으로 real하게 오르는가") 달성 = **예(usable는 아직 아님)**.
  다음 레버(인코더 fine-tune, 더 나은 pair)는 유보. fusion/rationale/RL은 계속 out of scope.

## Phase 1 진단 ① — oracle ceiling (2026-07-10): 데이터 모호성 기각, 병목은 텍스트 표현

Gate 4의 낮은 절대 retrieval이 데이터 벽인지 인코더 벽인지 심볼릭 상한/하한으로 판별
(`scripts/oracle_ceiling.py`, Gate 4와 동일 pool·strict tie):
- **oracle 상한(수+플래그를 완벽 전달 가정): 전 family R@50=1.0, R@10 0.91** → pool은 수
  수준에서 모호하지 않다. **데이터 모호성 가설 기각** (move-conditioned 57.4%에 대해).
- **무학습 mention baseline(SAN/UCI 문자열 매칭): MRR 0.0656, R@50 0.556** — 학습된 connector
  (0.0125 / 0.077)를 5.2×/7.2× 압도. **병목 = MiniLM이 move 토큰 식별 정보를 버리는 것.**
- 캐비엇: mention 신호는 심볼릭이지 언어 이해가 아니며(mate family의 UCI는 생성 산물),
  move 비언급 ~43%(gameknot·waterhorse)는 이 레버로 구제 불가.
- **재우선순위**: ① hybrid 텍스트 표현(문장 임베딩+move-mention 특징) → ② text encoder
  fine-tune → board encoder fine-tune은 근거 없음으로 강등. 상세: [scaleup_log.md](scaleup_log.md).

## Gate 5 (hybrid 심볼릭 특징, 2026-07-10) — 통과 (조건부)

진단 ①의 처방 실행: board (fen,target_move)→심볼릭 벡터[140], text 코멘트 파싱→mention
벡터[333]을 임베딩에 결합(hybrid) 또는 단독 사용(symbolic-only). 평가 장치 전부 동결 재사용.
- **symbolic-only 3-seed: t2b MRR 0.2795±0.0035, R@10 0.506, R@50 0.567** (Gate 4 대비
  MRR 22×, R@10 24×; frozen-probe 대비 26×). b2t 0.365. **usable top-k 최초 달성.**
- 9개 런(hybrid 6 + symbolic 3) 전부 global·within-family null 양방향 통과 — shortcut 아님.
- mate family는 oracle 상한 도달(R@50 1.000). 학습이 무학습 mention baseline도 4.3× 상회.
- fusion 발견: naive concat은 심볼릭 채널을 절반 희석하나 move 비언급 family에선 최선 —
  다음 개선은 score-level fusion/gating.
- 캐비엇: **심볼릭 신호의 회수이지 언어 이해의 증명 아님.** move 비언급 ~43% 세그먼트는
  여전히 R@50 ~0.15 — 의미 정렬의 남은 전선. fusion/rationale/RL은 계속 out of scope.

## 참조

- 결과물(gitignore): `outputs/scale_v1/**` (origin 로컬, robocopy 대피본).
- 커밋: 3da396b·44c821e·1fe7fec·aa582b6·e96486f·b461753·b6a2c67·fd63e8a·5bb712f·f8821bc·c177ba6.
- 요약(기계판독): [scale_v1_summary.json](scale_v1_summary.json).
