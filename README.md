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
python -m modalchess.eval.eval_baseline --config configs/eval/default.yaml --checkpoint outputs/train/model.pt
python -m modalchess.train.train_spatial_baseline --config configs/train/fen_baseline.yaml
python -m modalchess.eval.eval_baseline --config configs/eval/fen_baseline.yaml --checkpoint outputs/train_fen/model.pt
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

`history_fens`를 제공하는 경우 마지막 항목은 반드시 현재 `fen`과 같아야 하며, 중간 전이도 합법적인 단일 수로 연결되는지 검증합니다. `legal_moves_uci`를 제공해도 학습에는 보드 기준으로 다시 계산한 합법 수를 사용하고, JSONL 값은 검증용으로만 다룹니다.

`concept_tags`와 `engine_eval_cp`는 선택적 auxiliary 라벨입니다. 필드가 아예 없는 샘플은 `0-label`로 간주하지 않고, 해당 손실에서 마스킹되어 제외됩니다.

spatial baseline과 FEN baseline은 모두 동일한 `meta_features` 스칼라 입력 경로를 가질 수 있으며, 기본 설정은 두 모델 모두 `context` pooled 표현을 헤드에 사용합니다. 필요하면 head별로 `board` 또는 `context` pooled 선택을 config에서 바꿔 ablation할 수 있습니다.

기본 학습 엔트리포인트는 본 학습 체크포인트를 overfit 루프로 추가 오염시키지 않습니다. overfit은 테스트나 별도 실험에서만 직접 호출하도록 분리했습니다.
