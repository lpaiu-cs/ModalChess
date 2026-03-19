# 실험 계획

## 목표

공간 보드 인코더와 factorized move policy가 신뢰할 수 있는 연구 기반이 되는지 검증합니다.

## 1주차

- 실제 JSONL 파일럿 데이터에서 spatial/FEN 학습 및 평가 파이프라인 검증
- `best checkpoint = lowest val target_move_nll` 고정
- Spatial baseline과 FEN baseline의 첫 supervised 비교
- run별 config 사본, checkpoint, metrics json, failure dump, git hash, seed, model type 저장
- subset metric 분리: `promotion`, `castling`, `en_passant`, `check_evasion`

주간 산출물:

- aggregate 결과표
- subset 결과표
- representative failure dump
- 주간 결론 문서

## 2주차

- action-space alignment 또는 pooling/relation-bias 재검토 실험
- 1주차 결과가 spatial 우세인지, subset 우세인지, 혹은 FEN 우세인지에 따라 분기
- 향후 future fusion/rationale 인터페이스와 연결될 loss/report 구조 정리

## 비교 실험 축

- `B1`: FEN/text baseline
- `B2`: spatial baseline
- `B3`: spatial + meta token
- `B4`: spatial + meta token + relation bias

리포트에는 aggregate metric뿐 아니라 sample-level failure dump도 함께 남기며, 주간 집계는 `python -m modalchess.eval.aggregate_week1 --input-root outputs/week1`로 재생성할 수 있습니다.

## 성공 기준

- 코덱 round-trip 통과
- 텐서 invariant 통과
- 모델 forward shape 검증 통과
- tiny overfit 실행 시 손실 감소
- 평가 결과가 split별 리포트와 주간 aggregate 표로 출력
