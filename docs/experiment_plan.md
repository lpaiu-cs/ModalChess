# 실험 계획

## 목표

공간 보드 인코더와 factorized move policy가 신뢰할 수 있는 연구 기반이 되는지 검증합니다.

## 1주차

- 스키마와 좌표 규약 확정
- FEN 및 이동 코덱 구현
- 텐서 코덱과 fixture 데이터셋 구현
- round-trip 및 invariant 테스트 작성

## 2주차

- 로컬 fixture 데이터로 spatial baseline 학습
- state-fidelity 및 move-quality 지표 평가
- relation bias와 loss weight에 대한 소형 ablation 실행
- 향후 작업을 위한 future fusion/rationale 인터페이스 안정화

## 성공 기준

- 코덱 round-trip 통과
- 텐서 invariant 통과
- 모델 forward shape 검증 통과
- tiny overfit 실행 시 손실 감소
- 평가 결과가 구조화된 리포트로 출력
