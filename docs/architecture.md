# 아키텍처

## 범위

이번 단계는 다음 요소를 갖춘 공간 체스 연구 코어를 구현합니다.

- 구조화된 `BoardState`와 `BoardMeta`
- 가역적인 FEN 및 이동 코덱
- fixture와 JSONL 연구 데이터셋을 분리하는 데이터 경로
- `[H, C, 8, 8]` 형태의 보드 텐서 코덱
- square-aware 인코더와 factorized move policy
- 비교축으로 사용할 FEN/text baseline
- state fidelity, legality, value, concept를 위한 보조 헤드
- CUDA-aware 지도학습 및 평가 스캐폴딩

## 좌표 규약

이동 코덱의 square 인덱스는 `python-chess`를 따릅니다.

- `a1 = 0`
- `h1 = 7`
- `a8 = 56`
- `h8 = 63`

텐서 좌표는 항상 백 기준 시점을 사용합니다.

- `row 0, col 0 = a8`
- `row 0, col 7 = h8`
- `row 7, col 0 = a1`
- `row 7, col 7 = h1`

왕복 변환 헬퍼는 `modalchess.utils.square_utils`에 있습니다.

## 입력 텐서 스키마

각 스냅샷은 18개 채널을 사용합니다.

1. 백 폰
2. 백 나이트
3. 백 비숍
4. 백 룩
5. 백 퀸
6. 백 킹
7. 흑 폰
8. 흑 나이트
9. 흑 비숍
10. 흑 룩
11. 흑 퀸
12. 흑 킹
13. 현재 차례
14. 백 킹사이드 캐슬링 권리
15. 백 퀸사이드 캐슬링 권리
16. 흑 킹사이드 캐슬링 권리
17. 흑 퀸사이드 캐슬링 권리
18. 앙파상 대상

히스토리는 `[H, C, 8, 8]` 형태로 표현합니다. 길이가 부족한 경우 왼쪽을 0으로 패딩해서 최신 상태가 항상 `H - 1` 인덱스에 오도록 맞춥니다.

## 모델

베이스라인 모델은 토큰화 직전까지 공간 구조를 유지합니다.

1. square별로 history 채널을 결합
2. 각 square를 토큰 임베딩으로 투영
3. 2D positional encoding 추가
4. transformer 스타일 spatial block 적용
5. 명시적인 예측 헤드 연결

선택형 relation bias는 플러그인 형태로 분리되어 있으며, 단순한 기하 관계를 표현합니다.

- 같은 랭크
- 같은 파일
- 같은 대각선
- 나이트 이동 오프셋
- 킹 인접성

## 헤드

- `PolicyFactorizedHead`: source logits `[B, 64]`, destination logits `[B, 64]`, promotion logits `[B, 5]`
- `StateProbeHead`: 보드 plane과 명시적 상태 메타데이터를 복원
- `LegalityHead`: promotion-aware 조밀한 `[64, 64, 5]` exact-action legality 텐서를 예측
- `ValueHead`: 스칼라 value를 추정
- `ConceptHead`: 설정 가능한 멀티라벨 concept를 예측

공간 인코더는 최종적으로 두 종류의 pooled 표현을 분리해 반환합니다.

- `board_pooled`: 64개 보드 square 토큰만 평균한 표현
- `context_pooled`: 보드 토큰과 meta token을 함께 평균한 표현

현재 baseline에서는 value/concept/policy promo/state meta 복원에 `context_pooled`를 명시적으로 사용합니다. 이 분리는 이후 pooling ablation과 meta token 수 변화에 대한 비교를 안정화하기 위한 것입니다.

## 비교 기준선

실험 비교축을 위해 FEN 문자열만 읽는 선형 baseline도 함께 제공합니다.

- 문자 단위 FEN 토크나이저
- Transformer 기반 시퀀스 인코더
- FEN 시퀀스에서 64개 square query를 뽑아 동일한 action/state 출력 공간으로 투영
- spatial 경로와 동일한 scalar `meta_features`를 별도 meta token으로 받아 공정성을 유지

이 baseline은 spatial 모델과 동일한 policy/state/value/concept 평가 경로를 공유합니다.

## 범위 경계

이번 단계에서는 다음을 구현하지 않습니다.

- 최소 인터페이스를 넘는 LLM fusion
- 최소 인터페이스를 넘는 rationale 생성
- RL 또는 self-play
- 분산 학습
