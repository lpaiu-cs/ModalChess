# Phase 2 사전 등록 — 시각 모달리티: LM이 8×8 보드 토큰으로 판을 본다

> **North star**: 성공 = 언어 모델이 8×8 공간 토큰을 통해 판을 "보고", 프로그램적으로
> 검증 가능한 근거 있는 답을 한다. **검색 지표·특정 코퍼스 맞히기는 목표가 아니다.**
>
> 이 문서는 **사전 등록(pre-registration)**이다. 게이트 기준·margin·사살 사다리는 결과
> 열람 전에 커밋되며, 이후 수정은 본문 변경이 아니라 "개정" 절 추가로만 한다.
> (Phase 1 lever ②b에서 실천한 방식의 전면화.)

## 0. 배경: Phase 1의 교훈과 이 피봇의 근거

- 원래 의도(AGENTS.md·architecture.md): 체스판을 8×8 공간 세계로 읽는 모달 인코더를
  만들고 **LLM에 이식**한다("modality alignment with an LLM"). 인코더는 완성·검증됨
  (Gate 1: top-1 0.475, legality AP 0.991; 64칸 토큰 출력·fusion 인터페이스 보유).
- Phase 1(스크랩 코멘트 retrieval 정렬)은 이 의도에서 이탈했다: (a) 앵커가 스크랩 코멘트,
  (b) 인터페이스가 pooled 1벡터(공간 구조 소실), (c) 프록시 지표(MRR)가 목표化.
  결론(비언급 세그먼트 data-bounded)은 그 코퍼스의 성질이지 모달리티 목표의 한계가 아님.
- Phase 2는 정렬 대상을 **FEN에서 프로그램 생성한 grounded QA**(무한·무노이즈·검증 가능)로,
  인터페이스를 **64칸 토큰 → projection → frozen LM**으로, 손실을 **생성(answer CE)**으로
  교체한다. 파생 체스 지식은 가드레일대로 **입력이 아니라 타깃·검증기로만** 쓴다.

## 1. 재발 방지 규칙 (제도화)

1. **레버 심사 2질문** — 모든 신규 레버는 착수 전 로그에 답을 적는다:
   (a) 이 레버는 모달리티(표현/인터페이스)를 개선하는가, 특정 지표를 개선하는가?
   (b) 이 레버 성공 시의 수치 상승이 north-star 없이도 설명 가능한가(=게이밍인가)?
2. **캐비엇의 게이트 내장** — 각 게이트 정의에 "증명하지 않는 것" 필드 필수. 통과 선언문은
   그 필드를 함께 인용해야 유효하다.
3. **사전 등록 전면화** — margin/사살 기준/레버 사다리/seed 규약을 결과 전 커밋.
4. **기판 고정** — 게이트 진행 중 LM·인코더·데이터 교체 금지(핀은 §7).

## 2. 클레임 사다리 (무엇을 주장할 수 있는가)

| 클레임 | 내용 | 판정 게이트 | 판정 비교 |
|---|---|---|---|
| C1 지각 배선 | 보드 토큰이 frozen LM까지 배선되어 판독 가능 | P1 | Board vs Blind (+null) |
| C2 인터페이스 우위 | 공간 토큰이 FEN 문자열보다 나은 LM 입력 | P1 | Board vs FEN-최강 |
| C3 인코더 가치 | 사전학습 인코더가 원시 plane 대비 기여 | P2 | Board vs RawBoard (T2+) |
| C4 OOD 전이 | 인간 해설 언어로의 전이 | P3 | 특성화(합불 없음) |

C1 성공의 공식 문구는 "지각 배선"이지 "이해"가 아니다. T1은 인코더가 state probe로 직접
학습한 내용이므로 C3의 증거가 될 수 없다(C3는 T2 이상에서만).

## 3. Arm 설계 (동일 기판, 입력만 교체)

| Arm | 입력 | 학습 파라미터 | 역할 |
|---|---|---|---|
| **Board** | frozen 인코더 64토큰[384] → MLP → LM | MLP(2층) | 본 실험 |
| **RawBoard** | 원시 plane 칸별 18dim → 동일 구조 MLP → LM | MLP(2층) | 인코더 가치 대조 |
| **Blind** | 학습형 상수 토큰 64개 | 토큰 64×d | 사전확률/누출 바닥 |
| **FEN-zs** | "FEN: ..." 텍스트 프롬프트 | 없음 | 무학습 강 베이스라인 |
| **FEN-soft** | FEN 텍스트 + 학습형 soft 토큰 64개 | 토큰 64×d | 학습형 FEN 최강 후보 |

- **해석 대수**: FEN 콘텐츠 가치 = FEN-soft − Blind; 보드토큰 콘텐츠 가치 = Board − Blind;
  인코더 가치 = Board − RawBoard; 인터페이스 우위 = Board vs max(FEN-zs, FEN-soft).
- MLP: LN → Linear(d_in→5120) → GELU → Linear(5120→2560) → RMS 캘리브레이션 스케일
  (LM 임베딩 평균 RMS로 정합; P0에서 상수 측정·기록). d_in: Board 384, RawBoard 18.
  파라미터가 FEN-soft(≈0.16M)보다 Board(≈15M)에 많으나, projection은 포지션별 콘텐츠
  변환만 가능하고 과제 사전확률은 인코딩 불가 — 과제 적응 용량의 공정 비교는 Blind가 담당.
- 템플릿 게이밍은 전 arm이 동일 질문 분포를 공유하므로 **arm 간 델타에서 상쇄**된다.
  절대값이 아니라 델타로만 주장한다.

## 4. 과제 명세 v1 (T1 지각 5 + T2 정적 관계 4)

전 과제: 후보 고정 집합 + 정답 균형(목표 분포 ±2pt) + 템플릿 ≥4종(그중 1종은 학습
완전 배제 = held-out template). 채점 = 후보 logprob 합 argmax(1차), 길이정규화(2차 보고).

| # | task | 질문(대표) | 후보 | 균형 목표 |
|---|---|---|---|---|
| T1-1 | piece_on_square | What is on {sq}? | 13 (empty+12) | 13클래스 균등 |
| T1-2 | king_square | On which square is the {color} king? | 8 (정답+인접2+동선2+무작위3) | 색 50:50 |
| T1-3 | side_to_move | Whose turn is it to move? | 2 | 50:50 |
| T1-4 | castling_right | Does {color} still have the right to castle {side}? | 2 | yes/no 50:50, 4조합 균등 |
| T1-5 | piece_count | How many {color} {piece}s are on the board? | 4 (0/1/2/3+) | 4클래스 균등 |
| T2-1 | square_attacked | Is {sq} attacked by {color}? | 2 | 50:50 |
| T2-2 | piece_defended | Is the {color} piece on {sq} defended by another {color} piece? | 3 (yes/no/no-such-piece) | 1/3씩 |
| T2-3 | is_check | Is the side to move currently in check? | 2 | 50:50 |
| T2-4 | piece_pinned | Is the piece on {sq} pinned to its own king? | 3 (yes/no/no-such-piece) | 1/3씩 |

- **거짓 전제**(no-such-piece)를 3-way 과제에 1/3 포함 — 질문의 전제 자체가 정보를 누출하지
  못하게 한다.
- T3(1수 동역학: legal_move / capture_result / check_after_move)는 P2에서 명세 확정.
- **생성기/검증기 독립**: `qa_generator`(표적 샘플링·질문 구성)와 `qa_verifier`(fen+params
  만으로 정답 재계산)는 별도 코드 경로(다른 python-chess API 조합). 검증 불일치 1건 = P0 실패.

## 5. 데이터

- 소스: D1 `data/pilot/real_v2_scale/supervised_{train,val,test}.jsonl` (2.0M rows,
  game_id 보유). **QA-test 포지션은 supervised_test에서만** — 인코더 학습(supervised_train)
  미노출 포지션으로 평가.
- 위생: 포지션 키 = FEN 4필드(배치/턴/캐슬/ep) 해시로 exact-dedup; QA 스플릿 간 game 단위
  disjoint를 빌드 시 검증·로그(위반 시 test 우선 제거).
- 규모 v1(제안, P0 통계 후 확정치 기록): train 120k / val 6k / test 12k items.
- 시드: 생성 20260710, 학습 스크리닝 11 → 확정 {11,17,23}.

## 6. 게이트 사다리 (사전 등록 기준)

### P0 — 데이터·주입 인프라
- 통과: (a) 생성 QA 전량 독립 검증 불일치 0건, (b) 과제별 답 분포 목표 ±2pt,
  (c) split 위생 검증 로그, (d) 주입 sanity — 캘리브레이션된 무작위 토큰 64개 주입 시
  teacher-forced NLL 유한·생성 비붕괴, RMS 상수 기록, (e) LM revision 해시 핀 기입.
- 증명하지 않는 것: 모델 능력 일체.

### P1 — 4-arm 학습, T1+T2 (본 게이트)
- **유효성(필수)**: V1 Board − Blind ≥ **+30pt** (T1 집계 정확도, held-out 포지션);
  V2 shuffled-board null — 보드-질문 짝 셔플 시 Board 정확도 ≤ Blind + 5pt;
  V3 스크리닝 seed 11 통과 시에만 3-seed {11,17,23} 확정, 헤드라인은 3-seed mean±std.
- **본 판정 M1 (C2)**: Board vs best(FEN-zs, FEN-soft) — T1+T2 집계 **오류율 상대 감소
  ≥25%**, held-out 포지션과 held-out 템플릿 **양쪽**에서, 3-seed 방향 일치 + 평균 충족.
- **사살 사다리**(레버당 착수 전 2질문 심사 기록):
  - V1 실패 → 교정 2회 예산(RMS 캘리브레이션 수정 / projection 심화) → 재실패 시
    "인터페이스 부전"으로 정직 보고·중단.
  - M1 실패 → 레버1: projection 심화(4층 또는 perceiver-resampler) → 레버2: LoRA r16을
    **양팔 동일하게** → 소진 시 **"이 규모 LM에서 공간 인터페이스 이득 없음"으로 종결**
    (논제 반증 — 유효한 Phase 2 결과).
- 증명하지 않는 것: "이해"·체스 실력·인코더 가치(T1은 RawBoard 대조 클레임 불가).
  T1 통과의 공식 문구는 "지각 배선 성공".

### P2 — 인코더 가치·동역학·강건성
- C3: Board vs RawBoard, **T2 집계 오류 상대 감소 ≥25%** → "사전학습 인코더 가치" 인정.
- T3 과제 확장(명세는 착수 전 본 문서 개정 절에 사전 등록).
- 제2 LM 교차: gemma-2-2b(임베딩 √d 스케일링 캘리브레이션 필수) 또는 transformers 업그레이드
  검증을 통과한 Qwen3.5-4B에서 M1 **방향** 재현.
- 증명하지 않는 것: 탐색·전술 능력.

### P3 — 인간 언어 OOD 특성화
- Phase 1의 인간 코멘트 코퍼스를 **평가 전용**으로 복귀(예: 코멘트 조건 QA, LM likelihood
  매칭). 통과/실패가 아닌 특성화 보고. 학습 앵커로의 재사용 금지.

## 7. 고정 핀 (기판)

| 요소 | 값 |
|---|---|
| LM | Qwen/Qwen3-4B-Instruct-2507, 로컬 `E:/models/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554` (P0에서 핀) |
| LM 선택 근거 | 현 스택(transformers 4.57.3)이 지원하는 최신 dense(`qwen3`); Qwen3.5-4B는 `qwen3_5` 미지원으로 P2 강건성 축으로 이월; fallback = 캐시된 Qwen2.5-3B-Instruct |
| 인코더 | Tier M G3 seed11 `outputs/scale_v1/official/tier_m_g3/seed11/best_grounding_model.pt` (grounding 선택 기준이 "보기" 목적에 정합; `best_policy`는 P2 강건성 축) |
| 인코더 출력 | `tokens` [64, 384] (pooled 금지 — 공간 구조 보존) |
| 정밀도 | LM bf16 frozen + gradient checkpointing, projection fp32 |
| 학습 | answer 토큰 CE만, AdamW lr 1e-3(전 arm 공통, 스윕 없음 — arm별 lr 최적화는 공정성 훼손), wd 0.01, warmup 5% cosine, epochs 2, early stop val 기준, grad clip 1.0 |
| 알려진 한계 | lr 민감성 미탐색(공정성 우선); 후보 logprob 합의 길이 편향(전 arm 공유로 델타 무영향) |

## 8. Out of scope (Phase 2에서도 금지)

- RL / self-play, 장문 rationale 생성, LM 전체 파인튜닝(LoRA는 사살 사다리 최후단에서
  양팔 동일 조건만), Stockfish/엔진 통합, retrieval 지표의 재목표화(참고 보고만).

## 9. 개정 이력

- **개정 1** (P0 착수 시, 어떤 데이터·결과 열람 전): T1-5 piece_count 후보
  {0,1,2,3+} → **{0,1,2 or more}**. 사유: 3+ 클래스는 승격(promotion) 의존이라 실전
  데이터에서 균형 ±2pt 충전이 구조적으로 불가(가용성 분석에 의한 사전 수정, 결과 무관).

- **개정 2** (P1 스크리닝 중, board arm 완료 후·나머지 arm 결과 열람 전): FEN 텍스트 arm의
  긴 시퀀스로 학습 어텐션 메모리가 O(len²)로 커져 VRAM 오버플로(44GB, shared 스필).
  **micro_batch_size=8 + grad accumulation 도입, 유효 배치는 16 그대로 유지.** LayerNorm은
  배치 독립·LM은 동결이라 micro-batch는 그래디언트 평균 단위에만 영향 → board arm(micro=16,
  이미 완료)과 수치적으로 동등, 재실행 불필요. 판정 대상 수치(유효 배치·lr·epochs) 불변.

## 9b. P1b 사전 등록 — 수렴 + 하이브리드 (사용자 질문 3건에서 파생)

P1 스크리닝(seed11)이 V1·M1 미달했으나 두 교란이 드러남: (A) 연속 토큰 arm(board/rawboard)이
epoch2에서 Δ+0.05로 미수렴(fen_soft Δ+0.02) → "FEN 우위"가 학습량 교란일 수 있음. (B) 진짜
질문은 "공간 단독 vs FEN"이 아니라 **"시각 채널이 FEN 위에 값을 더하는가"**.

**P1b 설계 (결과 열람 전 등록)**:
- arms: **fen_soft**(FEN 단독), **board**(공간 단독), **hybrid**(FEN 텍스트 + board 토큰 동시).
  blind(2ep, 수렴)·fen_zs(무학습)·rawboard(2ep, C3 완료)는 참조로 재사용.
- epochs **4** (수렴 교란 A 대응 — best-val 추적으로 과적합 방지). 그 외 전부 P1과 동일
  (유효 배치 16·micro 8·lr 1e-3·동일 QA·동일 null). 기판 핀 불변.
- **사전 등록 예측 (반증 가능)**:
  - H-Q2: board@4ep가 fen_soft@4ep와의 격차를 좁히면 P1의 "FEN 우위"는 부분적으로 학습량
    아티팩트. board가 fen을 **따라잡거나 넘으면** → 공간 단독도 충분한 학습에서 경쟁력.
  - H-Q3 (핵심): **hybrid > fen_soft** (특히 is_check·pin에서) → 시각 채널이 FEN 위에
    값을 더함 = 모달리티 정당화. 판정선: hybrid가 fen_soft 대비 **is_check에서 +0.10 이상**
    이면서 overall ≥ fen_soft → "시각 채널 additive" 인정. hybrid ≈ fen_soft(전 과제)면
    → 시각 채널은 FEN이 있으면 불필요(모달리티 반증 강화).
  - 실패/종결: hybrid가 fen_soft를 어디서도 유의하게 못 넘고 board도 못 따라잡으면 →
    "이 LM·이 과제에서 시각 채널은 FEN 대비 additive 가치 없음"으로 종결(정직한 부정).
- 통과 시 3-seed 확정. QA 누출(king_square·attacked·defended)은 별도 트랙(V1 정화용),
  M1/hybrid 비교엔 상쇄되므로 P1b 판정엔 영향 없음.

## 10. 예산 추정

- QA 생성: CPU 수 분~수십 분. P1 학습: arm당 seed당 ~1.5h (RTX 5090, 4B frozen,
  120k items × 2ep) → 스크리닝 4 arm ≈ 6h, 확정 3-seed는 통과 arm 위주.
