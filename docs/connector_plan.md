# connector_v1 설계·구현 계획

- 상태: **구현·실행 완료 (2026-07-10). Gate 4 = 부분 통과.** 결과는 scaleup_log.md 참조.
  요약: within-family null 9/9 통과(shortcut 아님), b2t 3.3× 상승, mean 1.73× — 정렬은 real·재현적,
  그러나 절대 retrieval(R@50 ~8%)은 usable top-k 미달. 다음 레버(인코더 fine-tune 등)는 유보.
- 상태(원안): proposed, rev2 — GPT 리뷰(2026-07-10) 반영
- 전제: [scale_v1_decision.md](scale_v1_decision.md) Gate 3 = GO(조건부)
- rev2 보강(shortcut 방어가 핵심): within-family hard negatives(`family_blocked`, m≥2),
  within-source-family permutation null, multi-positive InfoNCE(false-negative group),
  early stop = mean(t2b,b2t), 다중지표 PASS(R@10/R@50/large-pool), linear projection baseline.
- 목표: **frozen board encoder + frozen 문장 인코더 위 작은 contrastive connector가
  board↔comment 정렬을 학습으로 frozen-probe 이상으로 올리는지 검증**. full fusion 아님.

## 범위 경계 (엄수)

- connector는 정렬 projection이지 fusion이 아니다. `future_fusion_stub.py`는 그대로 둔다.
- full LLM fusion / rationale generation training / RL은 여전히 out of scope.
- 두 인코더는 **frozen**. connector만 학습. (인코더 fine-tune은 명시적으로 다음 단계로 유보.)

## 핵심 설계

### 1. precompute-then-train
두 인코더가 frozen이므로 임베딩은 고정 → 한 번 precompute 후 connector는 그 벡터 위에서만 학습.
스텝당 비용이 MLP 2회 + NxN matmul뿐이라 큰 batch·다수 epoch·ablation grid가 싸다.
- board: Tier M `board_pooled`/`context_pooled` 384-dim (gate2_comment export 재사용).
- text: all-MiniLM-L6-v2 mean-pool 384-dim (comment_text, 1회 계산·캐시).
- 정렬 키: `probe_id`.

### 2. 모델 (~0.5–1M params)
- board head: `Linear(384→512) → GELU → Dropout → Linear(512→d) → L2norm`.
- text head: 동형(384→512→d).
- 공유 차원 `d ∈ {128, 256}` (ablation). 학습형 온도 `logit_scale`(CLIP식, init log(1/0.07), clamp).

### 3. 손실: multi-positive symmetric InfoNCE
`S = (Zb @ Ztᵀ) * exp(logit_scale)`; symmetric CE, in-batch negatives.
단순 diagonal-only CE는 **false negative**를 만든다 — 같은 `normalized_comment_text`(generic
코멘트가 여러 board에 붙음) 또는 같은 `position_id`(한 board에 여러 코멘트)는 사실상 같은
positive cluster다. 따라서 batch 내 false-negative group을 만들고 **multi-positive InfoNCE**
(그룹 내 임의 원소를 정답으로 인정)로 학습한다. 그룹 키: `normalized_comment_text` + `position_id`.
(주의: "same move"만으로 묶지 않는다 — 다른 board의 같은 수는 다른 positive다.)

### 4. 배칭 (불균형이 핵심 리스크)
train family 분포: mate_both_pairwise 42.9%, gameknot 19%, mate_testset 10.2%, … (504 family).
- **함정**: 순진한 "family 균등(family당 1개)"은 오히려 source_family를 **완벽한 shortcut**으로
  만든다 — batch 내 각 family가 1개뿐이면 "같은 family 찾기"로 정답이 풀린다.
- **올바른 설계 = `family_blocked`**: batch = `F families × m samples`, **m ≥ 2**(기본 m=4)로
  **같은 family 안 hard negatives를 반드시** 만든다 → 모델이 family/style이 아니라 **같은 family
  안에서 board를 구분**하게 강제.
- **min-family-size 처리**: singleton/소형 family(<m)는 blocked batch에서 제외하고 별도 `misc`
  pool로 섞어 넣는다(전량 버리지 않음). blocked 대상은 `≥m` 샘플 family.
- ablation `balance ∈ {family_blocked, family_capped, none}`.

### 5. 학습
AdamW + warmup_cosine, dropout. **early stop = `mean(strict_t2b_mrr, strict_b2t_mrr)`**
(또는 둘 중 낮은 쪽 포함) — t2b만 보면 기존 비대칭을 강화하고 Gate 4의 b2t 조건과 어긋난다.
30k train이 작으니 정규화·조기중단 중시. backbone **G3 기본 / G1 control**, seeds 11/17/23.
pool board 기본/context ablation.

## 평가 = frozen apparatus 재사용

connector projection 출력을 gate2 기계에 그대로 태운다:
- strict MRR/**R@10·R@50** 양방향 (3000행 정렬 test pool).
- **두 종류 permutation null 모두**: `global_shuffle` + **`within_source_family_shuffle`**.
  global만 이기고 within-family를 못 이기면 board semantics가 아니라 family/style을 맞춘 것이다.
  (within-family null은 `≥k` 샘플 family에만 적용 — tail singleton은 셔플 불가.)
- **source-holdout**: 일부 family로 학습, held-out family에서 retrieval (정렬이 미지 출처로 전이되나).
  week16/18 holdout regime 재사용 — held-out source의 b2t가 진짜 신호.
- large-pool shared queries (week18식) — 소수 후보에서만 오르는 착시 배제.
- **비교 기준선**: frozen-probe new+sentence t2b 0.01084 (반드시 이걸 명확히 넘어야 의미).

## Gate 4 (kill criteria, 명시)

PASS는 단일 지표가 아니라 **다중 조건**이다 (MRR만 오르고 top-k가 그대로면 실용 개선 아님):
- **최소선(kill 회피)**: `mean(t2b,b2t)` strict MRR이 frozen-probe 0.01084를 명확히 상회
  (예 ≥1.3×, seed noise 밖). 1.3×(~0.014)는 성공이 아니라 "죽이지 않을 하한"일 뿐.
- **실질 PASS**: 위 + **R@10·R@50 유의미 상승** + **large-pool에서도 유지** +
  **global·within-family null 둘 다 상회** + **board→text가 null 위로**.
- **HOLD/KILL**: connector ≈ frozen-probe(projection이 regression 대비 이득 없음), **또는**
  within-family null을 못 이김(= family/style shortcut), **또는** held-out source에서 b2t가 계속
  null 근처 → **data alignment 근본 상한**. 구조를 키우기 전에 멈추고 재검토(더 나은 pair,
  move-conditioned 필터, 또는 board→comment가 본질적으로 약하다는 결론 수용).

## 구현 단계

1. `src/modalchess/align/text_embed.py` — MiniLM mean-pool 래퍼(gate2_null_control 함수 공유 리팩터) + 캐시.
2. `src/modalchess/align/dataset.py` — precompute board+text .pt를 probe_id로 정렬한 pair dataset
   + family-balanced sampler + dup-text 마스킹.
3. `src/modalchess/align/connector.py` — projection heads + symmetric InfoNCE + 학습형 온도.
4. `src/modalchess/align/train_connector.py` — 학습 루프(AdamW/cosine/early-stop), connector+metrics 저장.
5. `src/modalchess/align/eval_connector.py` — gate2 strict-MRR + null + source-holdout로 채점, 기준선 대비.
6. `configs/connector/connector_v1.yaml`.
7. `tests/test_connector.py` — InfoNCE diagonal 회복(분리형 toy에서 loss→0·R@1→1), projection shape,
   balanced sampler 분포, dup-mask 정확성.
8. G3+G1 × 3 seed + ablation grid 실행 → Gate 4 판정 → decision 문서 갱신.

## Ablation grid (precompute라 저렴)

`projection ∈ {linear, mlp}`(비선형 head가 정말 필요한가 — linear baseline 필수)
× `d ∈ {128,256}` × `balance ∈ {family_blocked, none}` × `pool ∈ {board, context}`
× `온도 {learnable, fixed}` × `multi-positive {on, off}`. 학습이 초 단위라 grid 전체 실행.

## 리스크

- board→text 근본 상한 잔존 가능 → Gate 4 조기중단이 포착.
- noisy/generic 코멘트로 InfoNCE 오염 → dup-mask + family balance.
- 30k train 과적합 → dropout·early stop, **held-out source 성적이 진짜 신호**.
- frozen 고정으로 fine-tune 불가(의도적). connector가 평탄하면 다음 단계(가벼운 인코더
  fine-tune)로 넘기되 그건 이 계획 범위 밖.

## 산출물

- 코드: `src/modalchess/align/**`, `configs/connector/**`, `tests/test_connector.py`.
- 결과(gitignore): `outputs/connector_v1/**`.
- 판정: 본 문서 상태 갱신 + [scale_v1_decision.md](scale_v1_decision.md) Gate 4 기록.
