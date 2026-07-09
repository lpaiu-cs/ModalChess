# connector_v1 설계·구현 계획

- 상태: proposed (미구현)
- 전제: [scale_v1_decision.md](scale_v1_decision.md) Gate 3 = GO(조건부)
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

### 3. 손실: symmetric InfoNCE
`S = (Zb @ Ztᵀ) * exp(logit_scale)`; `L = ½(CE(S, arange) + CE(Sᵀ, arange))`. in-batch negatives.

### 4. 배칭 (불균형이 핵심 리스크)
train family 분포: mate_both_pairwise 42.9%, gameknot 19%, mate_testset 10.2%, … (504 family).
그대로 두면 negative가 동일 family로 쏠려 **board 정렬 대신 family/style 군집으로 치팅**한다.
- **source-family balanced sampler**: batch를 family ~균등(family당 cap)으로 구성.
  `balance ∈ {family_capped, family_uniform, none}` ablation.
- **duplicate-text negative 마스킹**: "good move"류 동일 코멘트가 서로 다른 board에 붙으면
  false negative가 된다. batch 내 동일 텍스트 쌍은 negative에서 제외(positive도 아님).
  `mask_duplicate_text_negatives: true`.

### 5. 학습
AdamW + warmup_cosine, dropout, **val t2b strict MRR 기준 early stop**. 30k train이 작으니
정규화·조기중단 중시. backbone **G3 기본 / G1 control**, seeds 11/17/23. pool board 기본/context ablation.

## 평가 = frozen apparatus 재사용

connector projection 출력을 gate2 기계에 그대로 태운다:
- strict MRR/R@K 양방향 (3000행 정렬 test pool).
- **permutation null control** — connector 출력이 null을 상회해야 함.
- **source-holdout**: 일부 family로 학습, held-out family에서 retrieval (정렬이 미지 출처로 전이되나).
  week16/18 holdout regime 재사용.
- large-pool shared queries (week18식) 보조 확인.
- **비교 기준선**: frozen-probe new+sentence t2b 0.01084 (반드시 이걸 명확히 넘어야 의미).

## Gate 4 (kill criteria, 명시)

- **PASS**: val t2b strict MRR이 frozen-probe 0.01084를 명확히 상회(예 ≥1.3×, seed noise 밖)
  **∧** permutation null 상회 **∧** board→text가 null 위로 올라옴.
- **HOLD/KILL**: connector ≈ frozen-probe(projection이 regression 대비 이득 없음),
  **또는** held-out source에서 board→text가 계속 null 근처 → **data alignment 근본 상한**.
  connector 구조를 키우기 전에 여기서 멈추고 재검토(더 나은 pair, move-conditioned 필터,
  또는 board→comment가 본질적으로 약하다는 결론 수용).

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

`d ∈ {128,256}` × `balance ∈ {family_capped, none}` × `pool ∈ {board, context}`
× `온도 {learnable, fixed}` × `dup-mask {on, off}`. 학습이 초 단위라 grid 전체 실행.

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
