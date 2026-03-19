# Pilot Preprocessing

이 디렉터리는 ModalChess 파일럿 데이터 전처리 결과와 관련 manifest를 보관한다.

## 지원 파이프라인

- `scripts/build_pilot_from_pgn.py`
  - Lichess PGN을 supervised backbone JSONL로 변환한다.
  - 출력은 `supervised_train.jsonl`, `supervised_val.jsonl`, `supervised_test.jsonl`이다.
- `scripts/build_puzzle_sidecar.py`
  - Lichess puzzle 원천을 `puzzle_eval.jsonl` 같은 sidecar 세트로 만든다.
  - 퍼즐 FEN은 첫 move 이전 상태이므로 `Moves[0]`를 먼저 적용한 뒤 `Moves[1]`를 target으로 잡는다.
- `scripts/enrich_with_lichess_eval.py`
  - supervised JSONL에 evaluation row를 조인한다.
  - 조인 키는 FEN의 첫 네 필드다.
- `scripts/build_mate_sidecar.py`
  - MATE 데이터를 language sidecar JSONL로 정규화한다.
  - 현재는 supervised target set으로 합치지 않는다.
- `scripts/normalize_chessgpt_corpus.py`
  - Waterhorse/chess_data를 text corpus와 conversation corpus로 분리 정규화한다.

## JSONL 계약

메인 supervised JSONL은 현재 ModalChess 계약을 따른다.

- `position_id`
- `game_id`
- `fen`
- `history_fens` (가능하면 포함)
- `target_move_uci`
- `next_fen`
- `concept_tags`
- `engine_eval_cp`
- `split`

핵심 검증 규칙:

- `history_fens`가 있으면 비어 있으면 안 된다.
- `history_fens[-1]`는 항상 현재 `fen`과 같아야 한다.
- 인접한 `history_fens`는 합법적인 단일 수 전이여야 한다.
- `next_fen`이 있으면 `target_move_uci`도 반드시 있어야 한다.
- supervised split은 `game_id` 단위로만 나눈다.

## 샘플 데이터

- raw 입력 예시는 `data/raw_samples/`에 있다.
- tiny sample 출력 예시는 `data/pilot/samples/`에 생성해 둘 수 있다.

## 데이터 적합성 메모

- Supervised backbone:
  - Lichess PGN
  - Lichess puzzles는 메인 backbone이 아니라 subset/special-rule eval 또는 augmentation 후보
- Language supervision:
  - MATE sidecar
  - Waterhorse normalized corpora
- External eval/value enrichment:
  - Lichess evaluation rows

## 권장 실행 예시

```bash
python scripts/build_pilot_from_pgn.py data/raw_samples/sample_games.pgn --output-dir data/pilot/samples --source-date 2026-03-19 --min-game-plies 1
python scripts/build_puzzle_sidecar.py data/raw_samples/sample_puzzles.csv --output-path data/pilot/samples/puzzle_eval.jsonl --source-date 2026-03-19
python scripts/enrich_with_lichess_eval.py --supervised data/pilot/samples/supervised_train.jsonl data/pilot/samples/supervised_val.jsonl data/pilot/samples/supervised_test.jsonl --evals data/raw_samples/sample_evals.jsonl --output-path data/pilot/samples/supervised_enriched.jsonl --source-date 2026-03-19
python scripts/build_mate_sidecar.py data/raw_samples/sample_mate.jsonl --output-path data/pilot/samples/language_mate.jsonl --source-date 2026-03-19
python scripts/normalize_chessgpt_corpus.py data/raw_samples/sample_chessgpt.jsonl --output-dir data/pilot/samples --source-date 2026-03-19
```
