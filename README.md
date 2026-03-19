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

JSONL 레코드는 최소한 `position_id`, `game_id`, `fen`을 포함해야 하며, 데이터셋 빌더는 `game_id` 기준으로 `train/val/test` split을 수행합니다.
