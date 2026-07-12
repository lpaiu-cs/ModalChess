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

## 인프라 사고·수정 (P1 스크리닝 1차 시도 중)

- **GPU 오버플로 진단**: board arm 프로세스 하나가 dedicated 30.8GB + **shared 7.4GB**(성능
  카운터로 확인) = ~38GB 사용, PCIe 스래싱. nvidia-smi의 dedicated만 보면 놓친다.
  근본 원인: `evaluate_items`가 전체 vocab(151,936) logits를 float 캐스트+log_softmax →
  ~3GB 임시 텐서가 캐싱 할당자 peak로 고정 → dedicated 한계 초과분이 shared로 스필.
  수정: 답 토큰 위치 슬라이스에만 log_softmax(수치 동일), del logits/embeds, eval 전
  empty_cache. peak reserved 38GB→22GB, 재실행 시 24GB·shared 정상.
- **체크포인트/재개 추가**: 하네스가 학습 종료 시에만 저장해, 1차 시도에서 board arm
  ~2h(epoch1 완료+epoch2 진입)를 kill로 유실. epoch 경계마다 `train_checkpoint.pt`
  원자적 저장(.tmp→replace) + train_arm 재개(마지막 완료 epoch 다음부터, optimizer/
  global_step/best/history 복원) + arm-level skip(eval.json 완료 arm 재실행 방지).
  재개 스모크 통과. → 이후 중단은 최대 1 epoch만 손실, 러너 재실행이 멱등.
- 후속 버그·수정: 체크포인트 저장 경로의 출력 디렉터리가 학습 후에야 생성돼 첫 epoch
  경계 저장이 크래시(board arm 재유실). `train_arm` 시작 시 mkdir로 수정, 미존재 디렉터리
  실CLI로 재검증. (교훈: 스모크가 디렉터리를 미리 만들어 버그를 가렸음 — 실행 조건 재현 필수.)

## P1 스크리닝 중간 결과 (seed 11)

### board arm (완료) — test n=11,693
- overall **0.8251** (T1 지각 0.8551 / T2 관계 0.7876), held-out 템플릿 0.7251,
  **shuffled-board null 0.5742**(틀린 보드 주입 시), best_epoch 2 val 0.8445.
- per-task: side_to_move 0.9992, is_check 0.9962, castling 0.9429, king_square 0.9154,
  piece_count 0.843, square_attacked 0.7746, piece_defended 0.7198, piece_pinned 0.6597,
  piece_on_square 0.5754(13-way, 최난).
- 예비 관찰: 보드 셔플 시 0.825→0.574로 25pt 하락 → 보드 콘텐츠가 실제로 쓰임(V2 예비
  지지). 단 사전 등록 판정(V1 vs Blind, M1 vs FEN-최강)은 해당 arm 완료 후. shuffled-null은
  틀린 보드를 여전히 주입하므로 Blind(보드 없음)와 다름 — 바닥값은 Blind로 확정해야 함.

### fen_soft arm (완료) vs board — M1의 핵심 접전
| metric | board | fen_soft | diff(F−B) |
|---|---|---|---|
| overall | 0.8251 | **0.8451** | +0.020 |
| T1 지각 | 0.8551 | **0.9293** | +0.074 |
| T2 관계 | **0.7876** | 0.7399 | −0.048 |
| held-out tpl | 0.7251 | **0.7825** | +0.058 |
| shuffled null | 0.5742 | 0.5986 | +0.025 |

per-task 핵심 반전: **piece_on_square** board 0.575 / fen_soft **0.789**(+0.213 — FEN은 판을
문자로 직접 나열하니 칸 읽기가 쉬움), **is_check** board **0.996** / fen_soft 0.721(−0.275 —
인코더는 state-probe로 in-check를 직접 학습해 보드 토큰에 체크 정보 내장, FEN엔 계산 필요).
king_square·piece_count·square_attacked는 fen_soft 우세(문자 조회), piece_pinned·defended 근소.

**예비 해석(seed11, Blind/rawboard 미완)**: 단순 논제("공간 토큰 > FEN 문자열")는 **지각
과제에선 반증 방향** — LM은 FEN 문자열을 매우 잘 읽는다. 공간 토큰이 이기는 건 인코더가
지도학습(state-probe)으로 **계산된 상태(체크 등)를 미리 담은** 과제뿐. 즉 M1(board가 FEN-최강을
오류 25%↓)은 **실패 방향**이고, 진짜 발견은 "사전학습 인코더의 가치는 raw 판독이 아니라
파생 상태에 있다"는 과제 의존적 분리 — 이는 C3(vs RawBoard)에서 정면 검증할 지점.

### 4-arm 정량 게이트 (seed11, test; rawboard 미완)
| arm | overall | T1 지각 | T2 관계 |
|---|---|---|---|
| board | 0.8251 | 0.8551 | 0.7876 |
| fen_soft | **0.8451** | **0.9293** | 0.7399 |
| fen_zs | 0.4463 | 0.4503 | 0.4413 |
| blind | 0.6460 | 0.6183 | 0.6806 |

- **V1 (Board−Blind, T1 ≥ +30pt): 미달** — +0.2368(23.7pt) < 30pt.
- **M1 (Board가 FEN-최강 오류 25%↓): 실패** — board 0.825 < fen_soft 0.845 (board가 뒤).
- **per-task Board−Blind**: side_to_move +0.475, is_check +0.474, castling +0.450,
  piece_count +0.275 (인코더 지도학습 속성 = board 압도) / king_square +0.042,
  piece_pinned +0.059 (근소) / **piece_defended −0.060, piece_on_square −0.057,
  square_attacked −0.045 (board가 blind보다 나쁨 — 공간 pooled 토큰이 세밀 판독엔 오히려 방해)**.
- **Blind 누출 경고(설계 성공)**: blind이 king_square 0.874·square_attacked 0.819·
  piece_defended 0.780로 높음 → 이 과제들은 질문 파라미터(칸·색)가 답과 상관되어 누출.
  깨끗한 과제(side_to_move·is_check·castling)는 blind ≈0.5로 board가 진짜 판독. V1의
  미달은 부분적으로 누출된 blind 바닥값 인플레 탓 — QA 누출 수정 시 Board−Blind 상승 여지.

**중간 결론(seed11)**: 사전 등록 두 정량 게이트 모두 미달 방향. 정직한 발견은 (a) FEN 문자열이
이 LM에는 더 나은 판 입력, (b) 공간 pooled 토큰은 **인코더가 지도학습으로 계산해둔 상태**
(턴/캐슬/체크/기물수)에서만 값을 더하고 세밀 판독은 오히려 약화, (c) blind가 QA 누출을 드러냄.
rawboard(C3)로 "그 파생-상태 이득이 사전학습 인코더 덕인지 raw plane으로도 되는지" 확정 예정.

## P1 스크리닝 최종 (seed11, 5-arm 완료) — 사전 등록 게이트 판정

| arm | overall | T1 지각 | T2 관계 | held-out | shuffled-null |
|---|---|---|---|---|---|
| fen_soft | **0.8451** | **0.9293** | 0.7399 | 0.7825 | 0.5986 |
| board | 0.8251 | 0.8551 | **0.7876** | 0.7251 | 0.5742 |
| rawboard | 0.8219 | 0.8918 | 0.7347 | 0.7437 | 0.6004 |
| blind | 0.6460 | 0.6183 | 0.6806 | 0.5837 | — |
| fen_zs | 0.4463 | 0.4503 | 0.4413 | 0.4806 | 0.3878 |

**게이트 판정**:
- **V1 (Board−Blind T1 ≥ +30pt): 미달** (+23.7pt). 단 blind이 QA 누출로 인플레(아래)이라
  깨끗한 과제에선 board가 진짜 판독 — 미달은 부분적으로 QA 아티팩트.
- **M1 (Board가 FEN-최강 오류 25%↓): 실패** — board 0.825 < fen_soft 0.845, board가 **뒤**.
  이 비교는 board·fen이 동일 QA를 보므로 누출이 상쇄됨 → **깨끗한 실패**.
- **C3 (board vs rawboard = 인코더 사전학습 가치): overall +0.003 (사실상 동률)**.
  per-task: **is_check board 0.996 vs raw 0.787 (+0.209)** · piece_pinned +0.062 (파생 상태)
  에서만 인코더 우위. raw 판독(piece_on_square raw 0.680 vs board 0.575 **−0.105**,
  king_square −0.042, piece_count −0.034, defended/attacked −0.03)은 **원시 plane이 더 나음**.

**최종 해석 (seed11 스크리닝)**: 사전 등록한 단순 논제 "공간 모달리티 > FEN 문자열"은
**반증**된다. (1) FEN 텍스트가 이 LM엔 더 나은 판 입력(M1 깨끗한 실패). (2) 사전학습 인코더는
원시 plane 대비 **전체 동률**이고 세밀 판독은 오히려 약화 — pooling이 crisp한 칸별 기물 정체성을
뭉갠다. (3) 인코더의 유일한 뚜렷한 가치는 **지도학습으로 미리 계산한 파생 상태(체크 +0.21, 핀)**
— 원시 plane도 FEN도 2 epoch 안에 못 만드는 것. (4) blind이 QA 누출(king_square·attacked·
defended에서 질문 파라미터가 답과 상관)을 드러냄 — 설계가 의도대로 작동, 단 V1 판정을 흐림.

**disposition**: 스크리닝이 V1·M1 미달 → 사전 등록상 3-seed 확정으로 자동 진행하지 않음.
M1 실패의 kill 사다리(deeper projection → LoRA)는 레버 심사 2질문 대상: rawboard≈board라
병목이 인코더/projection 깊이가 아니라 "연속 보드 토큰 < 텍스트(읽기)"라는 근본이므로, 깊은
projection이 M1을 뒤집을 개연성 낮음. 데이터가 가리키는 방향 = **하이브리드**(FEN이 읽기,
인코더 토큰이 계산된 전술: 체크/핀/위협)로 질문 재구성. 사용자 결정 필요(§ 사용자 보고).

## P1b (seed11, epochs 4) — 수렴 + 하이브리드: H-Q3 통과 (시각 채널 additive)

수렴 교란 제거(4ep) 후 fen_soft 재측정: overall 0.8451→**0.8664**(FEN도 학습량↑에 상승,
is_check 0.721→0.769). board@4ep(H-Q2)은 진행 중.

### H-Q3: hybrid(FEN+시각) vs fen_soft(FEN 단독), 둘 다 4ep
| task | fen_soft | hybrid | diff(H−F) |
|---|---|---|---|
| overall | 0.8664 | **0.8860** | **+0.020** |
| T1 지각 | **0.9543** | 0.9278 | −0.027 |
| T2 관계 | 0.7566 | **0.8338** | **+0.077** |
| **is_check** | 0.7692 | **0.9985** | **+0.229** |
| piece_pinned | 0.6751 | 0.7360 | +0.061 |
| piece_defended | 0.7598 | 0.7752 | +0.015 |
| square_attacked | 0.8223 | 0.8254 | +0.003 |
| king_square | 0.9885 | 0.9969 | +0.009 |
| castling | 0.9923 | 0.9931 | +0.001 |
| side_to_move | 1.0000 | 0.9892 | −0.011 |
| piece_count | 0.9661 | 0.9261 | −0.040 |
| piece_on_square | 0.8246 | 0.7338 | −0.091 |

**판정: H-Q3 통과** (사전 등록선: hybrid가 is_check +0.10↑ & overall ≥ fen_soft). 실제
is_check **+0.229**, overall +0.020 — **시각 채널이 FEN 위에 additive 가치를 준다**.
- 메커니즘: 시각 토큰은 FEN이 명시 안 하고 frozen LM이 FEN에서 계산 못 하는 **파생 상태**
  (체크 0.77→0.9985, 핀, T2 관계 전반)를 공급. board 단독의 is_check 0.996을 hybrid가 회수.
- **비균일**: 시각 채널은 raw 판독은 오히려 해침(piece_on_square −0.091, piece_count −0.040)
  — pooled 토큰이 깨끗한 텍스트보다 노이지. net +0.020은 T2 이득이 T1 손실을 상회한 결과.
- **캐비엇**: (1) 용량차(hybrid projection 15M vs fen_soft soft 164K) — 단 is_check 격차는
  용량이 아니라 정보 내용(FEN엔 체크 부재; board 단독도 동일 projection으로 0.996). (2) 이
  파생-상태 우위는 인코더의 지도학습 probe(체크·핀이 학습 타깃)에서 유래 = "raw 시각"이
  아니라 **학습된 특징의 전이**. in-check는 입력이 아니라 인코더 표현으로 전달되므로 가드레일
  누출 아님. 정확한 표현: "시각 모달이 판을 더 잘 본다"가 아니라 "인코더가 계산한 파생 상태가
  텍스트를 보강한다".

**P1→P1b 종합 재판정**: "공간 단독 > FEN"은 반증(P1) 유지. 그러나 사용자 Q3대로
**"공간 채널이 FEN을 보강하는가"는 YES** — 특히 계산된 상태에서 결정적. 모달리티의 값은
대체(substitute)가 아니라 **보강(augmentation)**에 있다.

### H-Q2: board@4ep vs fen@4ep (학습량 교란)
| | board@2ep | board@4ep | fen@4ep |
|---|---|---|---|
| test overall | 0.8251 | 0.8405 | 0.8664 |
| val overall | 0.8445 | **0.8670** | 0.8660 |

- **부분 확인**: 학습량↑에 board도 상승, val에선 fen과 동률(0.867≈0.866). 그러나 **test에선
  여전히 −0.026 뒤짐**. board는 val-test 격차 0.027(과적합·일반화 약함), fen은 격차 ~0.
  → "FEN 우위"는 순수 학습량 아티팩트가 아니라 **작지만 실재하는 판독 일반화 우위**.
  단 그 −0.026은 전부 raw 판독 과제 탓(piece_on_square board 0.66 vs fen 0.82)이고,
  board는 관계(T2 0.776 vs 0.757)·체크(0.998 vs 0.769)에선 fen을 앞선다.

### P1b 최종 (seed11, 3-arm @4ep)
| task | board | fen_soft | **hybrid** | blind(2ep) |
|---|---|---|---|---|
| overall | 0.8405 | 0.8664 | **0.8860** | 0.6460 |
| T1 지각 | 0.8924 | 0.9543 | 0.9278 | 0.6183 |
| T2 관계 | 0.7757 | 0.7566 | **0.8338** | 0.6806 |
| is_check | 0.9977 | 0.7692 | **0.9985** | 0.5223 |
| piece_on_square | 0.6608 | **0.8246** | 0.7338 | 0.6323 |

**hybrid이 세 arm 중 최고(0.886)** — FEN의 읽기 + 인코더의 파생 상태를 결합. board(0.840)·
fen(0.866) 모두 상회. V1 재계산(board@4ep−blind, T1)은 +0.274로 여전히 <+0.30(누출된 blind
기준)이나 2ep(+0.237)보다 개선.

**최종 결론(seed11)**: (1) 순수 대체 논제 반증 유지. (2) **보강 논제 확증 — hybrid > FEN 단독,
계산된 상태(체크·핀·관계)에서 결정적.** (3) 인코더 가치의 정확한 소재 = 지도학습으로 계산한
파생 상태의 텍스트 보강. **disposition: H-Q3 통과 → 3-seed 확정 대상 (사용자 결정).**

## P1c (seed11, 4ep, qa_v2=T1+T2+T3) — 하이브리드의 가치 경계 확정

사용자 선택(계산된 상태 심화)으로 T3(1수 동역학) 추가. **예상 밖: FEN이 move_gives_check를
0.858로 이미 잘함** — frozen LM이 FEN에서 1수 앞 체크를 상당히 추론(사전 등록 예측 "FEN 낮음"
빗나감).

### hybrid vs fen_soft @4ep (qa_v2)
| | fen_soft | hybrid | diff(H−F) |
|---|---|---|---|
| overall | 0.8571 | 0.8655 | +0.008 |
| T1 지각 | 0.9406 | 0.9169 | −0.024 |
| T2 정적관계 | 0.7666 | **0.8270** | **+0.060** |
| **T3 동역학** | **0.8387** | 0.8310 | **−0.008** |
| is_check (정적) | 0.8415 | **0.9954** | **+0.154** |
| piece_pinned (정적) | 0.6513 | 0.7390 | +0.088 |
| move_gives_check (동역학) | 0.8577 | 0.8515 | −0.006 |
| move_is_legal (동역학) | 0.8246 | 0.7885 | −0.036 |
| move_is_capture (동역학) | 0.8338 | 0.8531 | +0.019 |
| piece_on_square (판독) | 0.7885 | 0.7292 | −0.059 |

### P1c 4-arm 최종 + blind floor
| tier/task | board | fen_soft | hybrid | **blind** |
|---|---|---|---|---|
| overall | 0.8419 | 0.8571 | 0.8655 | 0.6890 |
| T1 지각 | 0.8830 | 0.9406 | 0.9169 | 0.6163 |
| T2 정적관계 | 0.8042 | 0.7666 | 0.8270 | 0.6720 |
| **T3 동역학** | 0.8236 | 0.8387 | 0.8310 | **0.8326** |
| move_gives_check | 0.8385 | 0.8577 | 0.8515 | **0.8462** |
| move_is_legal | 0.7977 | 0.8246 | 0.7885 | **0.8346** |
| move_is_capture | 0.8346 | 0.8338 | 0.8531 | 0.8169 |
| is_check(정적,대조) | 0.9977 | 0.8415 | 0.9954 | 0.5038 |

**⚠ 정정 (blind floor가 T3 비교를 무효화)**: 앞서 "동역학엔 전이 안 됨(인코더가 post-move
못 봄)"으로 썼으나, **blind T3=0.833으로 board 보는 어느 arm도 이를 못 넘음**(is_legal은
blind이 최고). **T3 과제는 수의 기하(frm→to 모양)만으로 대부분 풀려 보드-독립적**이다
(illegal move 생성이 기하 패턴으로 판별 가능한 설계 결함). 완벽한 판독기라도 board-independent
과제에선 blind 못 넘음 → **T3는 "시각이 동역학을 돕는가"를 판정 불가, 비교 자체 무효.**
blind arm이 이를 검출(T1/T2 누출 검출과 동일 — 사전 등록에 blind 넣은 이유).

**살아남는 견고한 발견 (blind로 검증됨)**:
- **정적 파생 상태 보강은 진짜**: is_check blind 0.504(우연) vs board/hybrid 0.995 — 거대한
  진짜 갭. pin도 동일 방향. **P1b 재현 + blind로 누출 아님 확증.**
- **판독 열세 재현**: piece_on_square hybrid<fen.
- **정밀화**: 시각/인코더 보강 가치 = 인코더 state-probe가 학습한 **정적 현재-국면 파생 상태
  (체크·핀)에 한정, 그리고 그것은 진짜**. 동역학은 **미판정(T3 재설계 필요 — 기하 누출 제거).**
