# scale_v1 실행 로그 (판정 기록)

> outputs/는 gitignore이므로, 판정에 사용된 핵심 수치와 결정을 이 문서에 남긴다.
> 원 아티팩트: `outputs/scale_v1/**` (로컬), git hash는 각 run_metadata.json에 기록됨.

## 2026-07-07 — W1: 인프라 + Tier S 스윕 + Elo A/B

### 인프라 (커밋 3da396b)
- lazy JSONL dataset + DataLoader 병렬화 + listwise 손실 벡터화 (~4× 학습 가속)
- 발견/수정: torch pin_memory가 배치 내 tuple을 list로 변환 → 튜플 membership 소비처
  전부 실패하는 잠복 계약 취약점. 소비처 정규화 + 회귀 테스트.

### 데이터 (2015-01 lichess 덤프, 전체 1,497,237 games)
- `real_v2_scale` (D1): 256,741 games / 2,000,000 positions (train 1.60M / val 199k / test 200k)
- `real_v2_scale_r1800` (D1r): 210,401 games / 1,659,769 positions — 양측 Elo≥1800 통과율 14.1%
- 두 빌드 모두 `--no-history` (H=1 프로토콜에서 파일/로드 비용 절감)

### LR 스윕 (Tier S d256/6L 5.14M params, bs512, 6 epochs, D1)
| LR | 궤적 | best val top-1 / NLL |
|---|---|---|
| 1e-3 | epoch 2부터 격렬 발산 (NLL 2.9→12.8) | 0.285 / 2.880 (ep1) |
| 3e-4 | 완만 발산 (train loss도 ep3부터 상승) | 0.404 / 1.979 (ep1) |
| **1e-4 + warmup 5%** | **깨끗한 수렴** | **0.401 / 1.946 (ep4)** |

- 판정: lr 1e-4 + warmup_cosine(0.05, min 0.05) 채택.
- 기존 백본(531k, 96k data, 2ep) 대비: top-1 0.318→0.401, top-5 0.66→0.80, NLL 2.41→1.95.
- honesty 관찰: raw legal_mass가 run/epoch에 따라 0.003~0.16으로 크게 요동 —
  G1(legality loss 없음)에서는 불안정한 진단 지표. G3에서 재평가 예정.

### VRAM 실측 (RTX 5090 32.6GB)
- 원인: forward의 pair/legality 경로 transient — Tier S bs512 train 10.9GB,
  eval bs1024 14.3GB; Tier L(d512/12L) bs256 train 13.5GB (bs512는 ~27GB로 불가).
- 대응: 사다리/공식 런 전체 bs256 + eval 256 통일, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- 참고: 학습 프로세스가 23.8GB(전용)+6.8GB(공유)까지 커밋해 스필오버 시 처리량 ~13% 저하 관찰.

### Elo A/B (Tier S, lr1e-4, 6 epochs) — 교차 평가 (top-1 / NLL)
| 학습→평가 | D1-val | r1800-val |
|---|---|---|
| D1 모델 | 0.4043 / 1.954 | 0.4170 / 1.903 |
| r1800 모델 | 0.3964 / 2.032 | 0.4319 / 1.859 |

- **판정: 주 데이터 축 = D1(무필터).** 근거: backbone 목적상 분포 일반화 우선 —
  D1 모델의 원정(r1800-val) 성적은 홈 대비 +1.3pt인 반면 r1800 모델의 원정은 −3.6pt.
  r1800-val은 라벨 품질 보조 지표로 게이트에 병기.
- 부수 발견: 두 모델 모두 고레이팅 수를 혼합 수보다 잘 예측 (강자 수가 더 규칙적).

### 진행 중
- 사다리: S256(d256, 20ep) → M(d384/8L, 20ep), bs256, val 50k — 완료 후 중단 규칙
  (개선 <2%면 상위 tier 중단) 적용해 L 투입 판단.

## 2026-07-08 — bf16 발산 근본 원인 확정 + Tier S 재확립

### listwise 발산의 근본 원인 (커밋 1fe7fec)
- 증상: bs256/20ep에서 S·M 모두 ep3부터 listwise 손실만 폭발 (axis CE·state probe는
  같은 인코더 위에서 계속 개선). LR이 감소하는 중에도 상승하는 비정형 패턴.
- 원인: pair scorer 로짓은 listwise CE 외에 제약이 없어 크기가 자라고, bf16(가수
  8bit)에서 큰 로짓의 gather+add 정밀도 손실이 기울기를 오염.
- 수정: listwise 점수 합산 fp32 고정 (수학 불변).
- 검증: 동일 seed 통제 비교 — bf16 ep6 NLL 6.25 폭발 vs **fp32 20ep 전 구간 단조
  개선**. 어제 bs512/6ep 런이 깨끗했던 것은 빠른 cosine 감쇠가 임계 크기 도달 전
  LR을 낮춘 우연.

### Tier S 확정 (d256/6L 5.14M, bs256, lr1e-4, 20ep, D1)
- **val top-1 0.4686 / top-5 0.8615 / NLL 1.6865** (best ep20, 여전히 개선 중)
- Gate 1 목표(top-1 ≥ 0.42, NLL ≤ 2.1) — **Tier S에서 이미 초과 달성**
- 기존 백본(531k/2ep) 대비 top-1 +15.1pt
- honesty(G1): illegal_top1 0.769, legal_mass 0.041
- 발산했던 bf16 런은 outputs/scale_v1/*_bf16diverged/로 보존

### 사다리 판정 (2026-07-08)
- Tier M (d384/8L, 15.0M): **top-1 0.4747 / NLL 1.6709** (best ep15, early stop ep19)
- S 대비 NLL 개선 0.93% (<2%) → **중단 규칙 발동, 모델 축 스케일업 종료. 병목은 데이터.**
- 결정: D2 확장 전에 Gate 2(표현 전이)를 먼저 측정 — Gate 1은 이미 S/M 모두 초과.
- 승자 tier = M. 공식 3-seed (G1+G3 × 11/17/23) 진행.

### Gate 2 미리보기 (2026-07-08, ladder-M seed11 G1, week7 프로토콜/language_probe_v2)
| MRR (구 백본 → 새 백본) | board→text | text→board |
|---|---|---|
| MATE (자연 텍스트) | 0.0005 → 0.0012 (~2.5×) | 0.0017 → 0.0057 (~3.3×) |
| Puzzle (합성 태그) | 0.059 → 0.098 (~1.7×) | 0.015 → 0.033 (~2.2×) |
- 8/8 구성 전부 개선 — H1 방향 지지. 절대값은 여전히 낮음(미약 신호 단계).
- 공식 Gate 2는 3-seed + G3 + week17/18 동결 comment regime + null control로 판정.
- 인프라: retrieval probe에 --backbone 필터 추가 (부분 backbone 실행 지원).

## 2026-07-09 — RAM 급증(error 1455) 진단 + 누수 방지

### 사건
- 공식 6런을 6-run 체인 cmd로 돌리던 중 g1 seed17에서 Windows `error code 1455`
  (ERROR_COMMITMENT_LIMIT, "paging file too small") 발생 — DataLoader 워커의
  `_share_filename_cpu_`에서 공유메모리 매핑 생성 실패. 메인은 죽은 워커를 기다리며
  6시간 행(zombie). 이후 사용자 리부트.

### 근본 원인 (3계층)
1. **Python 코드: 누수 없음.** epoch_metrics/prediction_rows/checkpoint payload 전부
   epoch 경계에서 해제되는 bounded 구조 (코드 감사 + 경험적 확증).
2. **DataLoader (주 기여): persistent_workers=True + pin_memory=True.** 워커가 20 epoch
   내내 생존하며 Windows 공유메모리 매핑을 해제 없이 유지, pinned 메모리가 non-pageable
   commit을 추가.
3. **시스템 취약점: commit limit 134GB** (RAM 126GB + 자동관리 페이지파일 ~8GB). 자동관리
   페이지파일이 급증 시 늦게 성장 → 사용자의 동시 대용량 작업(python 39 proc)과 겹칠 때
   commit 순간 초과.

### 대책 (코드 in-band, 커밋 예정)
- M 공식 config: `persistent_workers=False` (워커 epoch마다 teardown → 공유메모리 매 경계
  강제 반환, 누적 원천 차단), `pin_memory=False` (non-pageable commit 제거),
  `num_workers 8→4`, `prefetch 4→2` (in-flight 매핑 32→8).
- **워커 수는 학습 결과 불변** (셔플 순서는 시드 고정, num_workers는 배치 조립 분담만).

### 경험적 확증 (g1 seed17 재실행, 메모리 샘플러 30s 간격)
- **우리 학습 트리 WorkingSet: 1.9 → 2.4GB, 3시간 평탄** — 누수 소멸 확인.
- 시스템 commit 34~44% 진동은 전부 사용자 동시 작업 (우리 기여 2.4GB 고정).
- 운영: 사용자 결정으로 동시 실행 유지 + 메모리 가드(우리트리>8GB 또는 commit>78%)로 감시.
  페이지파일 확대는 보류 (코드 대책으로 충분 판정).

### 공식 6런 방식 전환
- 30h 체인 → **1런씩 분리 실행** (single_run.cmd <backbone> <seed>). 리부트/세션 종료 강건성.
- 1/6 완료: G1 seed11 top-1 0.4752 / NLL 1.6711 (재현성 확인, honesty legal_mass 0.135).

## 2026-07-10 — 공식 3-seed 완료 + Gate 1 통과 + worktree 삭제 복구

### 공식 6런 결과 (Tier M d384/8L, bs256, lr1e-4, fp32 listwise, D1, workers4/non-persistent)
| run | best ep | top-1 | top-5 | NLL | legality AP | legal_mass |
|---|---|---|---|---|---|---|
| G1 seed11 | 15/19 | 0.4752 | 0.8631 | 1.6711 | 0.0008 | 0.135 |
| G1 seed17 | 13/17 | 0.4758 | 0.8643 | 1.6671 | 0.0327 | 0.077 |
| G1 seed23 | 13/17 | 0.4728 | 0.8633 | 1.6724 | 0.0054 | 0.093 |
| G3 seed11 | 16/20 | 0.4750 | 0.8643 | 1.6712 | 0.9917 | 0.082 |
| G3 seed17 | 15/19 | 0.4748 | 0.8645 | 1.6714 | 0.9901 | 0.055 |
| G3 seed23 | 14/18 | 0.4743 | 0.8642 | 1.6732 | 0.9915 | 0.075 |

- **G1 평균: top-1 0.4746±0.0013, NLL 1.670, legality AP 0.013** (legality 감독 없음, 예상대로 0)
- **G3 평균: top-1 0.4747±0.0003, NLL 1.672, legality AP 0.9911** (정책 저하 0, 합법성 거의 완벽)

### Gate 1 판정: 통과 (3-seed, 모든 기준 큰 폭 초과)
- top-1 0.475 ≥ 목표 0.42 ✓ | NLL 1.67 ≤ 2.1 ✓ | G3 legality AP 0.991 ≥ 0.60 ✓
- 시드 분산 극소(±0.0003~0.0013) → 재현성 확정
- 기존 백본(531k/2ep) 대비: top-1 0.318→0.475, NLL 2.41→1.67, legality AP 0.415→0.991
- **다음: Gate 2 공식 판정** (6 checkpoint 임베딩 재수출 → week17/18 동결 comment regime + language_probe_v2 retrieval + null control)

### worktree 삭제 복구 (동일 날짜)
- worktree(claude/frosty-nightingale-74a0f3)가 git에서 분리됨. 결과물(outputs/scale_v1 9GB, gitignore)과
  학습데이터(real_v2_scale/r1800)는 worktree 로컬에만 존재 → origin으로 robocopy 대피 완료.
- 코드 7커밋은 브랜치에 안전. origin에서 `scale_v1` 브랜치 신규 생성해 복구(비파괴적).
- 교훈: gitignore된 대용량 산출물은 worktree 밖(origin outputs/)에 두거나 주기적 대피 필요.

## 2026-07-10 — Gate 2 (language_probe_v2 regime, 3-seed) 측정

### 새 백본(공식 Tier M) vs 기존 백본(week7), retrieval MRR, seed 11/17/23 평균
| family | 방향 | 기존 | 새 백본 | 배수 | 무작위 대비 |
|---|---|---|---|---|---|
| MATE(자연텍스트) | board→text | 0.0005 | 0.0014 | 2.7× | 4× |
| MATE(자연텍스트) | text→board | 0.0016 | 0.0059 | 3.7× | **12×** |
| puzzle(합성태그) | board→text | 0.055 | 0.099 | 1.8× | (tie 포함) |
| puzzle(합성태그) | text→board | 0.013 | 0.033 | 2.5× | **15×** |

- 무작위 기준선: MATE(N≈34,852) MRR 0.00032·R@1 0.000029, puzzle(N≈5,035) MRR 0.00181.
- strict(중복 tie 제외) 기준 새 백본 text→board R@1: MATE 0.00103(**35× 무작위**), puzzle 0.00877(**44× 무작위**).
- **16개 구성(2 backbone × 2 pool × 2 probe × 2 direction) 전부 기존 대비 개선**, 3-seed 재현.
- G1 vs G3: retrieval 거의 동일(legality 감독은 언어 전이에 영향 없음), puzzle에서 G3 미세 우위.
- 비대칭: text→board가 board→text보다 3~4× 강함.

### 해석 (정직)
- **신호는 진짜다**: 무작위 대비 12~44×, 3-seed 재현. 18주간의 "잡음"이 스케일업으로 실제 신호가 됨 — H1 방향 확증.
- **하지만 절대 수준은 여전히 약함**: MATE t2b MRR 0.006은 정답이 34k 후보 중 ~수백 등. 강한 retrieval(top-10)엔 못 미침.
- 병목이 백본 단독이 아님을 시사: 스케일업으로 크게 개선됐으나, 텍스트 축(tf-idf BoW)과 board↔comment 정렬 난이도가 남은 상한.

### 남은 공식 Gate 2 (미실행)
- week17/18 동결 comment regime(annotated_sidecar_eval_v6/holdout_v2) + 정식 permutation null control로
  comment 텍스트에서의 판정. 본 language_probe_v2 결과는 보조 확증(자연텍스트=MATE, 합성태그=puzzle).

## 2026-07-10 — 공식 Gate 2: comment regime + permutation null control

### 방법
- week17 고정 stratified subset(current_mixed_baseline/dedup, test 3000행)을 기존 임베딩 probe_id로
  정렬 재현 → 새·기존 백본이 동일 probe_id에 정렬(apples-to-apples).
- `scripts/gate2_null_control.py`: 동결 raw_text_retrieval probe 내부 함수 재사용, test 정렬을 50회
  무작위 치환한 permutation null과 real strict MRR 비교. 2 pool × 3 seed × 2 backbone.

### 결과 (comment regime, 12 config/그룹)
| 방향 | OLD real | NEW real | new/old | new / null-mean |
|---|---|---|---|---|
| text→board | 0.00559 | 0.00743 | 1.33× | 2.63× |
| board→text | 0.00317 | 0.00336 | 1.06× | 1.50× |

- null mean ≈ 0.0028(t2b) / 0.0022(b2t), null max ≈ 0.0047(t2b) / 0.0029(b2t).
- **NEW text→board: 12개 config 전부 real(min 0.00626) > null max(0.00472)** — 우연 아님 확정.
- board→text는 양쪽 다 null 근접(marginal) — 실질 신호 거의 없음, 개선도 없음.

### Gate 2 판정: 부분 통과 (H1 확증, 그러나 백본 단독 병목 아님)
- ✓ 신호 진짜: permutation null을 명확히 상회(comment + language_probe 두 regime, text→board).
- ✓ 새 백본이 기존 상회, 재현적: comment 1.33×, MATE/puzzle 2~3.7×.
- ✗ 절대값 여전히 약함, board→text는 near-chance, comment 개선폭 modest.
- 해석: 스케일업이 신호를 null 위로 밀어올려 H1(백본 병목) 확증. 그러나 텍스트 축(tf-idf BoW)과
  board↔comment 정렬 난이도가 남은 상한 — 특히 board→text.

### Gate 3 권고
- 바로 connector 학습으로 가기엔 절대 신호 부족. 최고가치 다음 실험 = **텍스트 축 교체(tf-idf →
  pretrained sentence encoder)**로 남은 병목이 텍스트 표현인지 검증. 상승하면 connector, 아니면
  board↔comment 정렬이 근본 한계.

## 2026-07-10 — 옵션 B: 텍스트 축 교체 (tf-idf → 문장 인코더) 2×2 요인 실험

### 방법
- 텍스트 target을 tf-idf BoW → `sentence-transformers/all-MiniLM-L6-v2`(384-dim, transformers mean-pool,
  normalized)로 교체. `scripts/gate2_null_control.py --text-side sentence`. probe·retrieval·permutation
  null은 동일. comment regime(정렬 3000행), 2 pool × 3 seed × 2 backbone.

### 결과 — text→board strict MRR (2×2 요인)
| | tf-idf 텍스트 | 문장인코더 텍스트 | 텍스트축 상승 |
|---|---|---|---|
| 기존 백본 | 0.00559 | 0.00692 | 1.24× |
| 새 백본 | 0.00743 | 0.01084 | 1.46× |
| 백본 상승 | 1.33× | 1.57× | |

기준선(기존 백본 + tf-idf = 0.00559)로부터 분해:
- 백본 스케일만: 0.00743 (1.33×)
- 텍스트축만: 0.00692 (1.24×)
- 둘 다: **0.01084 (1.94× total)** — 두 축이 곱셈적으로 복합(1.33×1.46 ≈ 1.24×1.57 ≈ 1.94).
- new+sentence: 12 config 전부 real(min 0.00945) > null max(0.00417), null mean 대비 3.82×.

### 판정
- **두 축(백본·텍스트)이 실재하고 대략 대등한 복합 병목**임을 요인 실험으로 확증. 어느 하나로도 불충분,
  함께면 신호 거의 2배. new+sentence는 프로젝트 사상 가장 깨끗한 above-chance board↔comment 신호.
- 정직 캐비엇: 절대 0.0108은 3000 후보 중 정답 ~90등 수준 — 여전히 usable retrieval(top-10) 미달.
  board↔comment 정렬에 근본 상한 가능성.
- **Gate 3 = GO(조건부)**: frozen-probe 0.0108은 학습형 connector의 하한. 두 축이 독립적으로 돕고
  복합하므로, 양쪽을 공동 최적화하는 small contrastive connector가 더 밀어올릴 근거 충분. 착수 권고.
  단 절대 정렬 상한을 조기 진단하는 체크포인트를 connector 초반에 둘 것.

## 2026-07-10 — connector_v1 구현 + Gate 4 판정 (부분 통과)

### 구현
- `src/modalchess/align/`: connector(multi-positive symmetric InfoNCE), dataset(family_blocked
  sampler m>=2 + min-family-size tail 처리), text_embed(MiniLM mean-pool 캐시), metrics(strict
  R@k + global·within-family permutation null), train/eval. tests/test_connector.py(9), 전체 122 통과.
- rev2 설계 반영: within-family hard negatives, ignore-mask(false negative), 2종 null,
  early stop=mean(t2b,b2t), linear baseline.

### 버그 → 진단 (3-seed 프로토콜이 잡음)
- 1차 grid에서 seeds 17/23이 null 아래로 붕괴. 진단: grid가 board 경로를 override하지 않아
  **모든 seed가 seed11 board로 학습**되고 eval만 per-seed board 사용 → 공간 불일치. train CLI에
  `--train/val/test-board` override 추가로 수정. (val은 건강한데 test만 붕괴 + real<null이 단서.)

### Gate 4 결과 (수정 후, comment regime 3000행 test, 3-seed)
| 구성 | t2b | b2t | mean | R@10 | R@50 | wf-null(t2b/b2t) |
|---|---|---|---|---|---|---|
| G3 mlp | 0.01254±0.0004 | 0.01315±0.0013 | 0.01284 | 0.021 | 0.077 | 3/3 · 3/3 |
| G1 mlp | 0.01170 | 0.01345 | 0.01257 | 0.020 | 0.083 | 3/3 · 3/3 |
| G3 linear | 0.01149 | 0.01202 | 0.01176 | 0.019 | 0.085 | 3/3 · 3/3 |

- vs frozen-probe(new+sentence): t2b 1.16×, **b2t 3.27×**, **mean 1.73×**.
- vs random(N=3000): MRR 4.4×, R@10 6.3×, R@50 4.6×.
- **9개 런(G1/G3/linear × 3seed) 전부 global·within-family null 양방향 통과** — family/style
  shortcut 아님이 재현적으로 확증. seed 분산 극소.
- linear vs mlp: mlp가 t2b +9%(0.0115→0.0125). 비선형 head 이득은 작고 linear도 전 null 통과.
- G1≈G3.

### Gate 4 판정: 부분 통과 (연구적 성공, 실용 미달)
- ✓ **학습형 board↔comment 정렬이 real·대칭·shortcut-robust·재현적**임을 최초 확증
  (within-family null 9/9). frozen-probe의 일방향(b2t near-chance) 약점을 connector가 해결(3.3×).
- ✓ min-bar(mean 1.73×) 통과.
- ✗ **절대 retrieval 여전히 약함**: R@50 ~8%(정답이 top-50에 8%) — usable top-k 미달. t2b는
  frozen-probe를 소폭만 상회.
- 해석: 정렬은 학습으로 오르고 real이지만, frozen-frozen contrastive의 modest 상한. 다음 레버
  (가벼운 인코더 fine-tune, 더 나은 move-conditioned pair)는 계획상 명시적 유보 — 본 단계 범위 밖.
- **결론**: "정렬이 학습으로 실제 오르는가?"에 **예(단, 아직 usable top-k는 아님)**. 최소 connector의
  목적을 달성했고 fusion/RL은 계속 out of scope.
