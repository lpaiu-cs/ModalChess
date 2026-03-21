# ModalChess

ModalChess는 구조화된 공간 체스 표현을 다루는 CUDA 대응 연구 코드베이스입니다.
이번 1단계는 다음에 집중합니다.

- 충실한 보드/상태 표현
- factorized move 코덱
- spatial baseline 모델링
- FEN/text baseline 비교축
- state-fidelity probe 및 평가 지표
- 향후 LLM fusion 및 RL 확장을 위한 명시적 인터페이스

베이스라인 입력은 원시 보드 상태로만 제한합니다.

- 12개 말 점유 plane
- 1개 선공 차례 plane
- 4개 캐슬링 권리 plane
- 1개 앙파상 대상 plane

파생된 체스 지식은 입력으로 넣지 않습니다.

## 빠른 시작

```bash
python -m pip install -e ".[dev]"
pytest
python -m modalchess.train.train_spatial_baseline --config configs/train/default.yaml
python -m modalchess.eval.eval_baseline --config configs/eval/default.yaml --checkpoint outputs/train/best_model.pt
python -m modalchess.train.train_spatial_baseline --config configs/train/fen_baseline.yaml
python -m modalchess.eval.eval_baseline --config configs/eval/fen_baseline.yaml --checkpoint outputs/train_fen/best_model.pt
python -m modalchess.train.train_spatial_baseline --config configs/train/pilot_spatial.yaml
python -m modalchess.eval.eval_baseline --config configs/eval/pilot_spatial.yaml --checkpoint outputs/week1/exp0_spatial_smoke/seed11/best_model.pt
python -m modalchess.train.train_spatial_baseline --config configs/train/pilot_fen.yaml
python -m modalchess.eval.eval_baseline --config configs/eval/pilot_fen.yaml --checkpoint outputs/week1/exp0_fen_smoke/seed11/best_model.pt
python -m modalchess.eval.aggregate_week1 --input-root outputs/week1
```

## 디렉터리 구성

- `src/modalchess/data`: 스키마, 코덱, fixture, 데이터셋 빌더
- `src/modalchess/models`: 보드 인코더, FEN baseline, 헤드, 코어 모델, 미래 확장용 스텁
- `src/modalchess/train`: 손실 함수, 옵티마이저, 트레이너, 학습 엔트리포인트
- `src/modalchess/eval`: 평가 지표, 리포트, 평가 엔트리포인트
- `src/modalchess/utils`: square 좌표, 디바이스/설정 유틸리티
- `docs`: 아키텍처, 데이터 스키마, 실험 계획, ablation 문서

## 데이터 입력

- `fixture`: 저장소 내부 smoke/회귀 검증용 소형 데이터
- `jsonl`: 실제 연구용 position 데이터 경로

JSONL 레코드는 최소한 `position_id`, `game_id`, `fen`을 포함해야 하며, 데이터셋 빌더는 `game_id` 기준으로 `train/val/test` split을 수행합니다. `split != all`인데 `game_id`가 없으면 기본적으로 에러를 내고, 정말 position-level split이 필요할 때만 `allow_position_level_split: true`를 명시적으로 켜야 합니다.

레코드에 `split` 필드가 모든 샘플에 일관되게 존재하면, 데이터셋 빌더는 ratio split 대신 그 값을 그대로 사용합니다. 1주차 파일럿 설정은 이 explicit split 경로를 사용합니다.

`history_fens`를 제공하는 경우 마지막 항목은 반드시 현재 `fen`과 같아야 하며, 중간 전이도 합법적인 단일 수로 연결되는지 검증합니다. `legal_moves_uci`를 제공해도 학습에는 보드 기준으로 다시 계산한 합법 수를 사용하고, JSONL 값은 검증용으로만 다룹니다.

`concept_tags`와 `engine_eval_cp`는 선택적 auxiliary 라벨입니다. 필드가 아예 없는 샘플은 `0-label`로 간주하지 않고, 해당 손실에서 마스킹되어 제외됩니다.

spatial baseline과 FEN baseline은 모두 동일한 `meta_features` 스칼라 입력 경로를 가질 수 있으며, 기본 설정은 두 모델 모두 `context` pooled 표현을 헤드에 사용합니다. 필요하면 head별로 `board` 또는 `context` pooled 선택을 config에서 바꿔 ablation할 수 있습니다.

기본 학습 엔트리포인트는 본 학습 체크포인트를 overfit 루프로 추가 오염시키지 않습니다. overfit은 테스트나 별도 실험에서만 직접 호출하도록 분리했습니다.

## 1주차 파일럿 규칙

- best checkpoint 선택 기준은 항상 `val target_move_nll` 최소입니다.
- 학습은 `best_model.pt`, `last_model.pt`, `train_metrics.json`, `run_metadata.json`, config 사본을 남깁니다.
- 평가는 `val/test` split별 `eval_report.*`, `eval_failures.jsonl`, 상위 `eval_summary.*`를 남깁니다.
- `python -m modalchess.eval.aggregate_week1 --input-root outputs/week1`로 run별 표와 subset 표를 다시 만들 수 있습니다.

`configs/train/pilot_*.yaml`과 `configs/eval/pilot_*.yaml`은 저장소 내부 `data/pilot/week1_fixture_pilot.jsonl`을 가리키는 smoke 설정입니다. 실제 파일럿 JSONL로 바꿔 쓸 때는 config를 수정하거나 CLI의 `--dataset-path`, `--train-dataset-path`, `--val-dataset-path`, `--output-dir`, `--seed` override를 사용할 수 있습니다.

## 데이터 전처리

원시 데이터에서 ModalChess JSONL을 만들 때는 `scripts/` 아래 전처리 엔트리포인트를 사용합니다.

- `scripts/build_pilot_from_pgn.py`
- `scripts/build_puzzle_sidecar.py`
- `scripts/enrich_with_lichess_eval.py`
- `scripts/build_mate_sidecar.py`
- `scripts/normalize_chessgpt_corpus.py`

샘플 raw 입력은 `data/raw_samples/`, 전처리 가이드는 `data/pilot/README.md`, 원천/라이선스 manifest는 `data/pilot/manifests/raw_sources.yaml`에 정리했습니다.

## 주차별 연구 흐름

- 1주차: real pilot supervised baseline 고정
- 2주차: action-space alignment (`axis_only < listwise < pair`)
- 3주차: grounding ablation (`G1 = policy_plus_state`, `G3 = policy_plus_state_plus_legality`)
- 4주차: frozen backbone 위에 language sidecar 정렬/스키마/embedding export 준비
- 5주차: standalone language probe corpora와 frozen-embedding readiness probe

현재 주 backbone 규칙은 다음과 같습니다.

- primary backbone: `G1`
- secondary control: `G3`
- checkpoint 선택 기준: 항상 `val.target_move_nll`

## 4주차 language sidecar 준비

4주차는 supervised backbone을 다시 학습하지 않고, 정렬 가능한 language sidecar와 rationale-ready schema를 준비합니다.

- alignment index: `scripts/build_language_alignment_index.py`
- rationale sidecar: `scripts/build_rationale_sidecar.py`
- QA report: `scripts/report_language_sidecar.py`
- frozen reference lock: `scripts/lock_week4_reference_artifacts.py`
- embedding export: `scripts/export_backbone_embeddings.py`

실행 순서는 보통 다음과 같습니다.

```bash
python scripts/build_language_alignment_index.py --supervised-train data/pilot/real_v1/supervised_train.jsonl --supervised-val data/pilot/real_v1/supervised_val.jsonl --supervised-test data/pilot/real_v1/supervised_test.jsonl --mate-path data/pilot/real_v1/language_mate.jsonl --puzzle-path data/pilot/real_v1/puzzle_eval.jsonl --output-root data/pilot/language_v1
python scripts/build_rationale_sidecar.py --input-root data/pilot/language_v1 --output-root data/pilot/language_v1
python scripts/report_language_sidecar.py --input-root data/pilot/language_v1 --output-dir data/pilot/language_v1/reports
python scripts/lock_week4_reference_artifacts.py --week3-root outputs/week3 --output-dir outputs/week4/reference_artifacts
python scripts/export_backbone_embeddings.py --checkpoint outputs/week3/exp3_ground_state/seed11/best_model.pt --output-dir outputs/week4/embedding_exports/g1/seed11 --dataset mate_val=data/pilot/language_v1/mate_matched_val.jsonl --dataset mate_test=data/pilot/language_v1/mate_matched_test.jsonl --dataset puzzle_val=data/pilot/language_v1/puzzle_matched_val.jsonl --dataset puzzle_test=data/pilot/language_v1/puzzle_matched_test.jsonl --dataset rationale_val=data/pilot/language_v1/rationale_val.jsonl --dataset rationale_test=data/pilot/language_v1/rationale_test.jsonl
```

정렬/스키마 계약은 `docs/language_sidecar_schema.md`에 정리되어 있습니다. 주의점은 다음과 같습니다.

- language sidecar는 supervised backbone train set에 합치지 않습니다.
- puzzle sidecar는 auxiliary eval/coverage 보강 용도입니다.
- ambiguous alignment는 강제로 병합하지 않고 unmatched로 남깁니다.
- week-5 readiness는 `data/pilot/language_v1/reports/language_sidecar_report.*`를 기준으로 판정합니다.

## 5주차 standalone language readiness probe

5주차는 exact matched-overlap을 억지로 늘리지 않고, standalone MATE/puzzle corpora 위에서 frozen G1/G3 임베딩의 language-adjacent signal을 점검합니다.

- standalone probe corpora: `scripts/build_probe_corpora.py`
- conservative target derivation: `scripts/build_probe_targets.py`
- probe/rationale QA report: `scripts/report_probe_corpora.py`
- frozen embedding export: `scripts/export_backbone_embeddings.py --format pt`
- linear/MLP readiness probes: `scripts/run_language_readiness_probes.py`

실행 순서는 보통 다음과 같습니다.

```bash
python scripts/build_probe_corpora.py --mate-path data/pilot/real_v1/language_mate.jsonl --puzzle-path data/pilot/real_v1/puzzle_eval.jsonl --output-root data/pilot/language_probe_v1
python scripts/build_probe_targets.py --input-root data/pilot/language_probe_v1 --output-root data/pilot/language_probe_v1
python scripts/report_probe_corpora.py --input-root data/pilot/language_probe_v1 --output-dir data/pilot/language_probe_v1/reports
python scripts/export_backbone_embeddings.py --checkpoint outputs/week3/exp3_ground_state/seed11/best_model.pt --output-dir outputs/week5/embedding_exports/g1/seed11 --format pt --dataset mate_train=data/pilot/language_probe_v1/mate_train.jsonl --dataset mate_val=data/pilot/language_probe_v1/mate_val.jsonl --dataset mate_test=data/pilot/language_probe_v1/mate_test.jsonl --dataset puzzle_train=data/pilot/language_probe_v1/puzzle_train.jsonl --dataset puzzle_val=data/pilot/language_probe_v1/puzzle_val.jsonl --dataset puzzle_test=data/pilot/language_probe_v1/puzzle_test.jsonl
python scripts/export_backbone_embeddings.py --checkpoint outputs/week3/exp3_ground_state_legality/seed11/best_model.pt --output-dir outputs/week5/embedding_exports/g3/seed11 --format pt --dataset mate_train=data/pilot/language_probe_v1/mate_train.jsonl --dataset mate_val=data/pilot/language_probe_v1/mate_val.jsonl --dataset mate_test=data/pilot/language_probe_v1/mate_test.jsonl --dataset puzzle_train=data/pilot/language_probe_v1/puzzle_train.jsonl --dataset puzzle_val=data/pilot/language_probe_v1/puzzle_val.jsonl --dataset puzzle_test=data/pilot/language_probe_v1/puzzle_test.jsonl
python scripts/run_language_readiness_probes.py --embedding-root outputs/week5/embedding_exports --target-root data/pilot/language_probe_v1 --output-dir outputs/week5/readiness_probes --backbone-seed 11
```

주의점은 다음과 같습니다.

- week-5는 backbone 재학습이 아니라 frozen-embedding probe 주간입니다.
- MATE `candidate_moves`는 single `target_move_uci`로 강제 변환하지 않습니다.
- puzzle는 text-free theme probe 성격이 강하고, MATE는 coarse text-derived tag probe 성격이 강합니다.
- week-6 진입 상태는 `outputs/week5/readiness_probes/*`와 `data/pilot/language_probe_v1/reports/*`를 함께 보고 판단합니다.

## 6주차 split-hygiene v2 및 multi-seed eval-only probe

6주차는 frozen backbone을 유지한 채, week-5 probe의 방법론 안정성을 강화합니다.

- probe corpora를 `data/pilot/language_probe_v2`로 다시 빌드합니다.
- split은 가능하면 `game_id` 그룹 기준을 우선 사용하고, 반복 그룹이 없으면 명시적으로 `source_row_id`로 fallback합니다.
- readiness probe는 G1/G3 모두 seed `11/17/23`로 다시 측정합니다.
- retrieval-style probe는 lexical bag-of-words 기준의 eval-only readiness check이며, language fusion 결과로 해석하지 않습니다.

실행 순서는 보통 다음과 같습니다.

```bash
python scripts/build_probe_corpora.py --mate-path data/pilot/real_v1/language_mate.jsonl --puzzle-path data/pilot/real_v1/puzzle_eval.jsonl --output-root data/pilot/language_probe_v2 --prefer-game-id-group-split --min-game-id-group-size 2
python scripts/build_probe_targets.py --input-root data/pilot/language_probe_v2 --output-root data/pilot/language_probe_v2 --rare-label-threshold 20
python scripts/report_probe_corpora.py --input-root data/pilot/language_probe_v2 --output-dir data/pilot/language_probe_v2/reports --compare-root data/pilot/language_probe_v1
python scripts/export_backbone_embeddings.py --checkpoint outputs/week3/exp3_ground_state/seed11/best_model.pt --output-dir outputs/week6/embedding_exports/g1/seed11 --format pt --dataset mate_train=data/pilot/language_probe_v2/mate_train.jsonl --dataset mate_val=data/pilot/language_probe_v2/mate_val.jsonl --dataset mate_test=data/pilot/language_probe_v2/mate_test.jsonl --dataset puzzle_train=data/pilot/language_probe_v2/puzzle_train.jsonl --dataset puzzle_val=data/pilot/language_probe_v2/puzzle_val.jsonl --dataset puzzle_test=data/pilot/language_probe_v2/puzzle_test.jsonl
python scripts/run_language_readiness_probes.py --embedding-root outputs/week6/embedding_exports --target-root data/pilot/language_probe_v2 --output-dir outputs/week6/readiness_probes --backbone-seed 11 --backbone-seed 17 --backbone-seed 23
python scripts/run_retrieval_readiness_probes.py --embedding-root outputs/week6/embedding_exports --corpus-root data/pilot/language_probe_v2 --target-root data/pilot/language_probe_v2 --output-dir outputs/week6/retrieval_probes --backbone-seed 11 --backbone-seed 17 --backbone-seed 23
```

주의점은 다음과 같습니다.

- week-6도 여전히 eval-only language work입니다. connector training, LLM fusion, rationale training은 하지 않습니다.
- split hygiene v2 report는 `data/pilot/language_probe_v2/reports/v1_vs_v2_split_diff.json`에 남깁니다.
- readiness aggregate는 `outputs/week6/readiness_probes/probe_results_aggregate.csv`에 남깁니다.
- retrieval probe는 `outputs/week6/retrieval_probes/*`에 저장되며, 반드시 retrieval-style readiness로만 해석합니다.

## 7주차 text-realism / source-audit / raw-text retrieval

7주차는 connector training 없이, week-6 probe가 실제 텍스트를 얼마나 다루고 있었는지 감사하고 raw-text retrieval을 추가합니다.

- week-6 reference lock: `outputs/week7/reference_from_week6/`
- text realism audit: `scripts/audit_probe_text_realism.py`
- auxiliary source audit/build: `scripts/audit_aux_language_sources.py`, `scripts/build_aux_language_corpora.py`
- MATE keyword densification audit: `scripts/audit_mate_keyword_coverage.py`
- raw-text retrieval probe: `scripts/run_raw_text_retrieval_probes.py`

실행 순서는 보통 다음과 같습니다.

```bash
python scripts/audit_probe_text_realism.py --input-root data/pilot/language_probe_v2 --week6-readiness-path outputs/week6/readiness_probes/probe_results.json --week6-retrieval-path outputs/week6/retrieval_probes/retrieval_results.json
python scripts/audit_aux_language_sources.py --output-dir data/pilot/language_probe_v3/reports
python scripts/build_aux_language_corpora.py --output-root data/pilot/language_probe_v3
python scripts/audit_mate_keyword_coverage.py --input-path data/pilot/real_v1/language_mate.jsonl --output-dir data/pilot/language_probe_v3/reports
python scripts/run_raw_text_retrieval_probes.py --embedding-root outputs/week6/embedding_exports --corpus-root data/pilot/language_probe_v2 --output-dir outputs/week7/raw_text_retrieval --backbone-seed 11 --backbone-seed 17 --backbone-seed 23
```

주의점은 다음과 같습니다.

- MATE retrieval은 실제 `strategy_text + tactic_text`를 사용합니다.
- puzzle retrieval은 natural text가 아니라 synthetic tag-string retrieval입니다.
- 둘은 같은 종류의 언어 평가가 아니므로 summary에서 분리해서 해석해야 합니다.
- week-7도 여전히 eval-only week입니다. connector training, LLM fusion, rationale training은 하지 않습니다.

## 8주차 split hygiene fix / auxiliary fetch / realism upgrade

8주차는 alignment training을 시작하지 않고, week-6/7에서 남아 있던 data realism과 split hygiene gap을 operationalize합니다.

- split-hygiene fix: `scripts/build_probe_corpora.py`, `src/modalchess/data/probe_corpora.py`
- auxiliary fetch lock: `scripts/fetch_aux_language_sources.py`
- file-level auxiliary normalization: `scripts/build_aux_language_corpora.py`
- target realism report + conservative MATE v2 targets: `scripts/report_target_realism.py`
- optional aux raw-text retrieval rerun: `scripts/run_raw_text_retrieval_probes.py --family aux_board_anchored`

실행 순서는 보통 다음과 같습니다.

```bash
python scripts/build_probe_corpora.py --mate-path data/pilot/real_v1/language_mate.jsonl --puzzle-path data/pilot/real_v1/puzzle_eval.jsonl --output-root data/pilot/language_probe_v3_fix --prefer-game-id-group-split --min-game-id-group-size 2 --report-output-dir data/pilot/language_probe_v3_fix/reports --compare-root data/pilot/language_probe_v2
python scripts/build_probe_targets.py --input-root data/pilot/language_probe_v3_fix --output-root data/pilot/language_probe_v3_fix
python scripts/fetch_aux_language_sources.py --manifest-path data/pilot/language_probe_v4/manifests/aux_fetch_lock.yaml --notes-path data/pilot/language_probe_v4/manifests/aux_fetch_notes.md
python scripts/build_aux_language_corpora.py --output-root data/pilot/language_probe_v4
python scripts/report_target_realism.py --probe-root data/pilot/language_probe_v3_fix --aux-root data/pilot/language_probe_v4 --mate-keyword-audit-path data/pilot/language_probe_v3/reports/mate_keyword_audit.json --output-root data/pilot/language_probe_v4
python scripts/export_backbone_embeddings.py --checkpoint outputs/week3/exp3_ground_state/seed11/best_model.pt --output-dir outputs/week8/embedding_exports/g1/seed11 --format pt --dataset aux_board_anchored_train=data/pilot/language_probe_v4/aux_board_anchored_train.jsonl --dataset aux_board_anchored_val=data/pilot/language_probe_v4/aux_board_anchored_val.jsonl --dataset aux_board_anchored_test=data/pilot/language_probe_v4/aux_board_anchored_test.jsonl
python scripts/run_raw_text_retrieval_probes.py --embedding-root outputs/week8/embedding_exports --corpus-root data/pilot/language_probe_v4 --output-dir outputs/week8/raw_text_retrieval_v2 --family aux_board_anchored --backbone-seed 11 --backbone-seed 17 --backbone-seed 23
```

주의점은 다음과 같습니다.

- `prefer_game_id_group_split`는 다시 실제 code path로 연결됐지만, 현재 MATE/puzzle probe corpora는 repeated `game_id`가 없어 여전히 `source_row_id` fallback이 맞습니다.
- Waterhorse snapshot은 bounded partial fetch입니다. week-8 기본 fetch는 `annotated_pgn`과 selected `dataset_info`만 받습니다.
- auxiliary normalization은 file-by-file로 진행하고, `board_anchored_text`, `text_only`, `unusable`를 섞지 않습니다.
- `mate_targets_v2`는 conservative audit 기반 부가 target set이며, week-6 target을 silently replace하지 않습니다.
- week-8 raw-text retrieval은 여전히 evaluation-only probe입니다. tiny frozen alignment readiness를 판단하는 근거이지, language fusion 결과가 아닙니다.
