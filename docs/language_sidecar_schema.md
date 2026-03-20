# Language Sidecar Schema

## Purpose

Week-4 does not train a language model. It freezes the supervised backbone and prepares split-safe language sidecars for week-5 alignment experiments.

The language layer stays separate from the supervised backbone:

- supervised train/eval continues to use the real-pilot PGN dataset
- language sidecars are matched by safe keys only
- ambiguous matches stay explicit or unmatched

## Alignment Index Schema

Aligned sidecar rows in `data/pilot/language_v1/*.jsonl` use this canonical schema.

- `sidecar_id`: stable sidecar row id
- `source`: source dataset name
- `source_row_id`: original row id or stable fallback
- `split`: matched supervised split, or `null` if unmatched/aux-only
- `matched_supervised`: whether the row matched a supervised position
- `matched_position_id`: supervised `position_id` if matched
- `matched_game_id`: supervised `game_id` if matched
- `fen`: sidecar board state
- `fen_4field`: FEN normalized to the first four fields
- `target_move_uci`: sidecar move target if the source defines one
- `candidate_moves`: optional candidate move list
- `strategy_text`: optional strategy text
- `tactic_text`: optional tactic text
- `theme_tags`: optional theme or concept tags
- `alignment_type`: one of:
  - `fen_exact`
  - `fen_exact_target_move`
  - `fen_4field`
  - `fen_4field_target_move`
  - `unmatched`
  - `aux_only`
- `alignment_confidence`: numeric confidence score
- `notes`: ambiguity or provenance note

Additional provenance fields may be attached when useful:

- `preferred_move`
- `history_fens`
- `matched_target_move_uci`
- `matched_next_fen`

## Matching Rules

Matching is conservative.

1. Try exact full-FEN matching first.
2. If exact FEN is ambiguous and the source semantics are verified, allow `(FEN, target_move_uci)` disambiguation.
3. If exact FEN fails, allow 4-field FEN matching.
4. If 4-field FEN is ambiguous and the source semantics are verified, allow `(FEN_4field, target_move_uci)` disambiguation.
5. If multiple supervised rows still remain, keep the sidecar row unmatched.

Current source policy:

- `MATE`: no target-move coercion; use FEN matching only
- `puzzle_eval`: move-conditioned disambiguation is allowed because `target_move_uci` is explicit
- `ChessGPT aux`: copied as auxiliary corpora without supervised matching

## Rationale-Ready Schema

Week-4 rationale rows in `data/pilot/language_v1/rationale_*.jsonl` use this schema.

- `rationale_id`: stable rationale row id
- `position_id`: matched supervised position id
- `matched_game_id`: matched supervised game id
- `fen`: board state
- `target_move_uci`: move-conditioned target
- `focus_squares`: key squares for the move or defense
- `focus_pieces`: pieces directly involved
- `motif_tags`: tactical or structural motif tags
- `threat_tags`: attack-side tags
- `defense_tags`: defense-side tags
- `king_safety_flag`: whether the current side is already in check
- `promotion_flag`
- `castling_flag`
- `en_passant_flag`
- `check_evasion_flag`
- `rationale_short`: short text only, no chain-of-thought
- `source`: sidecar provenance
- `source_confidence`: inherited alignment confidence

Extra provenance is preserved:

- `split`
- `sidecar_id`
- `source_row_id`
- `alignment_type`
- `theme_tags`

## Embedding Export Schema

Frozen embedding exports in `outputs/week4/embedding_exports/*/*.jsonl` contain:

- `position_id`
- `split`
- `source`
- `target_move_uci`
- `board_pooled`
- `context_pooled`
- `checkpoint_path`
- `seed`
- `git_hash`
- `model_parameter_count`
- `sidecar_id`
- `alignment_type`

Optional:

- `square_tokens`

## Provenance Rules

- Never collapse ambiguous many-to-one or one-to-many matches into fake clean labels.
- Preserve source row ids and source names in every derived file.
- Preserve supervised split integrity in all matched outputs.
- Keep puzzle sidecars out of supervised training.
- Keep language sidecars out of the supervised backbone training set.
