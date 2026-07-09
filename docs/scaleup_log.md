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
