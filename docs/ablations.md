# Ablation 계획

이번 단계의 ablation은 의도적으로 작게 유지하고, 아키텍처 검증에 집중합니다.

## 권장 ablation

- relation bias 사용 여부
- history length `H=1` 대 `H=2`
- legality loss weight 사용 여부
- state probe loss weight 사용 여부
- policy head의 pair scorer 비활성화 대 활성화

## 평가 관점

이동 품질과 상태 충실도를 함께 추적합니다.

- top-1 및 top-k 합법 수 정확도
- 말 점유 복원 정확도
- 차례 예측 정확도
- 캐슬링 권리 정확도
- 앙파상 정확도
- 체크 상태 정확도
- legality 행렬 정확도

## 비목표

- 엔진 강도 최적화
- 탐색 또는 RL 비교
- LLM fusion ablation
