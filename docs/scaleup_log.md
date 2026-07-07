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
