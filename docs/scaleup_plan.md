# scale_v1: Backbone 스케일업 계획과 런북

## 배경과 가설

week18 판정(`STILL_EVAL_ONLY_BUT_STABLE`)의 근본 원인 가설:

**H1. 언어 신호 부재의 주원인은 corpus 품질이 아니라 backbone의 용량·학습량 부족이다.**

근거: 기존 G1/G3 backbone은 531k 파라미터(d_model=128, 2-layer)를 96k 포지션에서
epochs=2로만 학습했고, 모든 seed에서 best epoch == last epoch(수렴 전 중단)였다.
4~18주차 전체가 이 frozen 임베딩 위에 구축되었다.

scale_v1은 G1/G3 프로토콜과 기존 평가 인프라를 유지한 채 데이터 ~20×, 모델
~9~70×, 학습량 ~10×를 올려 week17/18의 동결 eval regime에서 before/after를 직접
비교한다. 비목표: 아키텍처 재설계, fusion 학습(게이트 통과 후), corpus 재정제.

## 데이터

- `data/pilot/real_v2_scale`: 2015-01 덤프, rated-only, ~2M positions, `--no-history`
- `data/pilot/real_v2_scale_r1800`: 동일 + `--min-rating 1800` (Elo A/B용)
- Elo A/B는 Tier S에서 1회 수행해 이후 데이터 축을 고정한다.

```bash
python scripts/build_pilot_from_pgn.py data/pilot/raw/lichess/standard/lichess_db_standard_rated_2015-01.pgn.zst \
  --output-dir data/pilot/real_v2_scale --rated-only \
  --min-game-plies 10 --max-game-plies 200 --max-ply-index 120 \
  --max-games 400000 --max-positions 2000000 --max-positions-per-game 8 \
  --sample-every-n-plies 4 --random-seed 11 --no-history
```

`--no-history`는 H=1 학습에서 history_fens 기록을 생략해 파일 크기와 로드 시
전이 검증 비용을 없앤다. 실측 처리량: ~184 games/s, ~7.75 positions/game
(RTX 5090 머신, 2M 빌드 ≈ 25분).

## 신규 학습 인프라 (이번에 추가됨)

- **lazy JSONL dataset** (`dataset_builder.LazyJsonlDataset`): 스캔 1회로 split
  hygiene 검증 + 바이트 offset 인덱스만 메모리에 두고 `__getitem__`에서 인코딩.
  dataset config 키: `loading: lazy`, `validate_sample_rate: 0.01`
  (표본 full 검증은 스캔 시점에 결정적 stride로 수행; 전수 검증은 빌드 단계 책임).
- **DataLoader 병렬화**: train config 키 `num_workers`, `pin_memory`,
  `persistent_workers`, `prefetch_factor`. collate는 Windows spawn 피클을 위해
  `functools.partial`로 교체됨.
- **LR 스케줄**: `train.lr_schedule: {name: warmup_cosine, warmup_ratio, min_lr_ratio}`
  (기본 constant). step 단위로 적용된다.
- **Early stopping**: `train.early_stop_patience: N` — val `target_move_nll`이
  N epoch 연속 개선되지 않으면 중단. best-checkpoint 선택 계약은 그대로다.
- **PGN 빌더**: `--min-rating` (양쪽 Elo 기준, 누락 시 탈락), `--no-history`.
- train epoch 로그에 `samples_per_second`, `learning_rate`가 기록된다.

## 스케일 사다리

| Tier | config | 구성 | ~params | seeds |
|---|---|---|---|---|
| S | `configs/train/scale_v1_g1_s.yaml` | d256/6L | ~5M | 11 |
| M | `configs/train/scale_v1_g1_m.yaml` | d384/8L | ~14M | 11 |
| L | `configs/train/scale_v1_g1_l.yaml` + `scale_v1_g3_l.yaml` | d512/12L | ~38M | 11/17/23 |

- Tier S에서 LR {1e-3, 3e-4} 및 Elo A/B(real_v2_scale vs real_v2_scale_r1800)를 판정.
- **사다리 중단 규칙**: tier를 올려도 val NLL 개선 < 2%면 모델 축 중단, 데이터
  축(D2: 추가 월 덤프)으로 전환.
- 공식 3-seed는 Tier 승자에서 G1+G3 페어로만 수행.

```bash
python -m modalchess.train.train_spatial_baseline --config configs/train/scale_v1_g1_s.yaml
python -m modalchess.eval.eval_baseline --config configs/eval/scale_v1.yaml \
  --checkpoint outputs/scale_v1/tier_s_g1/seed11/best_model.pt \
  --output-dir outputs/scale_v1/tier_s_g1/seed11/eval
```

eval config는 model 구성을 비워 checkpoint의 `resolved_model_config`를 사용한다.

## 게이트

**Gate 1 — 백본 품질 (Tier L 3-seed):**
val top-1(legal-conditioned) ≥ 0.42 (기존 0.318), NLL ≤ 2.1 (기존 2.41),
occupied ≥ 0.99 유지, G3 legality AP ≥ 0.60 (기존 0.415).
honesty 지표(raw legal_mass, illegal_top_1_rate)는 게이트가 아니라 진단으로 보고.

**Gate 2 — 표현 전이:**
`export_backbone_embeddings.py`로 새 checkpoint 임베딩 재수출 후 동결된
week17/18 regime(`annotated_sidecar_eval_v6`, `holdout_v2`)과 `language_probe_v2`
에서 readiness/retrieval probe 재실행. 기준: strict MRR이 null control CI 밖
**그리고** 기존 backbone(0.006) 대비 ≥ 3×.

**Gate 3 — 판정:**
- 통과 → tiny contrastive connector 학습 착수 (eval-only 탈출)
- 실패 → H1 기각, 병목은 텍스트 표현(tf-idf) → pretrained sentence encoder 교체 실험
- 어느 쪽이든 corpus 재정제로 돌아가지 않는다.
- 판정은 `outputs/scale_v1/decision.md`에 기록하고 요약 md/json은 저장소에 커밋한다
  (기존 outputs/ 미보존 문제 재발 방지).

## 리스크 메모

- 2015-01 덤프 전체 ~1M games. min-rating 1800 통과율이 낮으면 r1800 빌드가 2M에
  못 미칠 수 있다 — A/B는 동일 positions 수로 서브샘플해 비교한다.
- GPU 사용률이 낮으면(< 70%) packed shard 경로를 추가한다 (조건부 워크스트림).
