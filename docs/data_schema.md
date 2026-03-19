# 데이터 스키마

## BoardMeta

`BoardMeta`는 보드 plane에 직접 들어가지 않는 메타데이터를 명시적으로 보관합니다.

- `side_to_move`
- 캐슬링 권리
- `en_passant_square`
- `halfmove_clock`
- `fullmove_number`

## BoardState

`BoardState`는 합법적인 체스 포지션을 다음 구조로 저장합니다.

- `pieces: dict[int, str]`
- `meta: BoardMeta`

말 정보의 key는 `python-chess` square 인덱스를 쓰고, value는 `P`, `n`, `k` 같은 FEN 심볼을 사용합니다.

이 표현은 다음을 만족하도록 설계되어 있습니다.

- FEN 복원
- `python-chess` 보드 재구성
- `python-chess`와 일치하는 합법 수 도출
- 공간 plane으로 보드 인코딩

## FactorizedMove

`FactorizedMove`는 다음 필드를 가집니다.

- `src_square: int`
- `dst_square: int`
- `promotion: int`

프로모션 어휘는 다음과 같습니다.

- `0 = none`
- `1 = knight`
- `2 = bishop`
- `3 = rook`
- `4 = queen`

## PositionSample

`PositionSample`은 fixture, 학습, 평가에 공통으로 쓰는 레코드입니다. 다음 정보를 포함합니다.

- 식별자와 FEN 필드
- `game_id`
- 히스토리 FEN 목록
- 사전 계산된 board plane
- `BoardState` 안의 명시적 `BoardMeta`
- 합법 수 목록
- 선택적 지도학습 타깃 수
- 선택적 다음 포지션
- concept 태그
- 선택적 스칼라 value proxy

## State probe 타깃

state probe는 현재 상태에 대한 타깃만 복원합니다.

- 12 current piece planes
- 현재 차례
- 4 castling rights
- 앙파상 square 또는 없음
- 체크 상태 라벨

이 값들은 타깃과 평가 지표로만 쓰이며, 모델 입력으로는 절대 사용하지 않습니다.

## JSONL 연구 데이터 인터페이스

실제 연구 데이터는 JSONL 한 줄당 하나의 position record를 가정합니다. 최소 권장 필드는 다음과 같습니다.

- `position_id`
- `game_id`
- `fen`
- `target_move_uci` (선택)
- `history_fens` (선택)
- `next_fen` (선택)
- `concept_tags` (선택)
- `engine_eval_cp` (선택)

학습/평가 split은 position이 아니라 `game_id` 기준으로 수행합니다.
