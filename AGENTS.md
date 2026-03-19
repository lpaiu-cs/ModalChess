# AGENTS.md

## Project
ModalChess

## Mission
Build the first two-week research core for ModalChess: a Python research codebase that is CUDA-ready, modular, testable, and designed around a structured spatial chess world-state.

This phase is not about SOTA strength.
This phase is about:
- correct data representation
- clean architecture
- reliable codecs
- spatial baseline modeling
- state-fidelity evaluation
- future extensibility toward language fusion and RL

## Research framing
ModalChess treats chess as a structured spatial world-state, not merely a linear text serialization.

Core hypothesis for this phase:
A spatial board representation with an explicit factorized move policy and state-grounded auxiliary heads can form a correct research substrate for later language-model integration.

Important:
The novelty is not "using 2D tensors" alone.
The codebase must be structured so later phases can test:
- spatial world-state grounding
- move-conditioned rationale generation
- spatial action masking in RL
- modality alignment with an LLM

## Scope for the first two weeks (must implement)
1. Repository skeleton and architecture docs
2. BoardState schema
3. FEN <-> BoardState codec
4. Tensor codec for board state -> board planes
5. UCI move <-> factorized move codec
6. Spatial board encoder
7. Factorized move policy head
8. Auxiliary heads:
   - state probe head
   - legality head
   - value head
   - concept head
9. Training/evaluation scaffolding
10. State fidelity metrics
11. Tiny synthetic dataset / fixture dataset for smoke tests
12. Interface stubs for future LLM fusion and rationale modules

## Out of scope for this phase (do not implement)
- full LLM fusion
- unrestricted chain-of-thought or long-form rationale generation
- RL / self-play / PPO
- distributed training or multi-node infrastructure
- UI, frontend, web demo
- search engine integration
- internet-dependent dataset download flows
- manual reimplementation of chess rules when python-chess already provides them

## Non-negotiable scientific guardrails
1. Do not feed derived chess knowledge as input features.
   The following are NOT valid model inputs in this phase:
   - legal move masks
   - attacked squares
   - in-check labels
   - mate threat labels
   - engine evaluations
   These may be used as:
   - targets
   - metrics
   - verifiers
   - later RL constraints

2. The board representation must be sufficient for legal play.
   Minimum per-snapshot board channels:
   - 12 piece occupancy planes
   - 1 side-to-move plane
   - 4 castling-right planes
   - 1 en-passant target plane
   Total minimum channels per snapshot: 18

3. Metadata that does not naturally live as board planes must remain explicit.
   Examples:
   - halfmove clock
   - fullmove number
   - repetition count if available

4. The move policy must be factorized:
   - source square
   - destination square
   - promotion

5. The factorized move codec must support all legal UCI moves, including promotions.

6. Rationales are not a full deliverable in this phase.
   Only define schemas/interfaces for future move-conditioned rationale modules.

7. Prefer clean, explicit research abstractions over premature optimization.

## Coordinate systems and indexing (must be explicit and tested)
There must be no ambiguity here.

### Move codec indexing
Use python-chess square indices for move codec logic:
- a1 = 0
- b1 = 1
- ...
- h8 = 63

### Tensor coordinate convention
For board planes, use tensor coordinates from White's perspective:
- row 0, col 0 = a8
- row 0, col 7 = h8
- row 7, col 0 = a1
- row 7, col 7 = h1

Provide helper utilities for:
- square index -> (row, col)
- (row, col) -> square index

These round-trips must be unit-tested.

## Promotion encoding
Use an explicit promotion vocabulary:
- 0 = none
- 1 = knight
- 2 = bishop
- 3 = rook
- 4 = queen

For non-promotion moves, promotion class must be `none`.

## Environment and runtime
Use:
- Python 3.11+
- PyTorch
- HuggingFace Transformers (future hooks only in this phase)
- python-chess
- pytest
- YAML-based configs

### CUDA requirements
- Default to CUDA when available for training.
- Maintain CPU fallback for tests and smoke runs.
- Do not hardcode `.cuda()` anywhere; use device-aware code.
- Use autocast on CUDA when safe.
- Prefer bf16 autocast when supported; otherwise use fp16 carefully.
- Keep single-GPU as the default.
- DDP and distributed orchestration are out of scope.

## Implementation style
- Use `src/` layout.
- Use type hints everywhere meaningful.
- Use dataclasses for structured records.
- Keep modules small and composable.
- Separate data, model, training, and evaluation logic.
- Prefer pure functions for codecs and conversion utilities.
- Avoid hidden global state.
- Avoid giant scripts with mixed responsibilities.
- Use config-driven experiment entry points.
- Use clear docstrings on public classes and functions.

## Repository layout to create
modalchess/
  pyproject.toml
  README.md
  AGENTS.md
  configs/
    baselines/
      spatial_policy.yaml
    model/
      board_encoder.yaml
      heads.yaml
    train/
      default.yaml
    eval/
      default.yaml
  docs/
    architecture.md
    data_schema.md
    experiment_plan.md
    ablations.md
  src/modalchess/
    __init__.py
    data/
      __init__.py
      schema.py
      board_state.py
      fen_codec.py
      tensor_codec.py
      move_codec.py
      fixtures.py
      dataset_builder.py
      collators.py
    models/
      __init__.py
      board_encoder.py
      spatial_positional_encoding.py
      relation_bias.py
      modalchess_core.py
      future_fusion_stub.py
      future_rationale_stub.py
      heads/
        __init__.py
        policy_factorized.py
        state_probe.py
        legality.py
        value.py
        concept.py
    train/
      __init__.py
      losses.py
      optim.py
      trainer.py
      train_spatial_baseline.py
    eval/
      __init__.py
      metrics_state_fidelity.py
      metrics_move_quality.py
      report.py
      eval_baseline.py
    utils/
      __init__.py
      device.py
      seed.py
      config.py
      logging.py
      square_utils.py
  tests/
    test_square_utils.py
    test_fen_codec.py
    test_tensor_codec.py
    test_move_codec.py
    test_board_encoder_shapes.py
    test_policy_factorized.py
    test_state_probe.py
    test_train_smoke.py
    test_eval_smoke.py

## Data layer requirements

### Core dataclasses
Implement explicit dataclasses for at least:
- BoardMeta
- BoardState
- FactorizedMove
- PositionSample

### BoardMeta
Include fields such as:
- side_to_move
- white_can_castle_kingside
- white_can_castle_queenside
- black_can_castle_kingside
- black_can_castle_queenside
- en_passant_square
- halfmove_clock
- fullmove_number

### BoardState
Must represent a single legal chess position faithfully enough to:
- reconstruct FEN
- produce legal moves through python-chess agreement
- encode to board planes

### FactorizedMove
Must contain:
- src_square: int
- dst_square: int
- promotion: int

### PositionSample
Must support training/evaluation pipelines and contain fields such as:
- position_id: str
- fen: str
- history_fens: list[str]
- board_planes: torch.Tensor        # [H, C, 8, 8]
- meta: dict[str, int | float | str | None]
- legal_moves_uci: list[str]
- target_move_uci: str | None
- next_fen: str | None
- concept_tags: list[str]
- engine_eval_cp: float | None

It is acceptable for some fields to be optional in fixture data.

## Tensor encoding requirements
Input board tensor shape must be:
- [H, C, 8, 8]

Rules:
- `H` = configurable history length
- `C` = minimum 18 channels per snapshot
- default simple runs may use `H=1`
- history support must exist in the interfaces, even if smoke fixtures are shallow

Do not flatten the board too early in data code.
Preserve spatial structure until the model boundary.

## Model architecture requirements

### Board encoder
Implement a spatial encoder that works on board planes and outputs square-aware latent representations.

Recommended minimal approach:
1. Concatenate history features per square
2. Convert each square to a token
3. Project square features to `d_model`
4. Add 2D positional encoding
5. Process with Transformer blocks

Required modules:
- `SquarePatchEmbed` or equivalent square-token embedding
- `SpatialPositionalEncoding2D`
- `BoardEncoder`

### Relation bias
Implement an optional relation-bias module as a separate component.
It should be pluggable and not tightly coupled.

Possible relation categories:
- same rank
- same file
- same diagonal
- knight offset
- king adjacency

Keep this optional for ablations.

### Core model
Implement a `ModalChessCoreModel` that combines:
- board encoder
- policy head
- auxiliary heads

This is the first-phase core.
Do not add LLM fusion in this phase beyond stubs/interfaces.

## Head requirements

### Policy head
Must be factorized and explicit.

Minimum outputs:
- src logits: [B, 64]
- dst logits: [B, 64]
- promo logits: [B, 5]

Optionally support a pair scorer module:
- biaffine or small MLP scorer over (src, dst)

Default legal-move scoring should be well-defined.
A recommended formula is:
score(move=u->v,p) = src_logit[u] + dst_logit[v] + promo_logit[p] + pair_score(u, v)

Legal move filtering is evaluation logic, not an input feature.

### State probe head
Implement a lightweight probe/reconstruction head from the encoder latent to:
- current piece planes
- side to move
- castling rights
- en-passant target (or none)
- in-check label

This is important for state-fidelity evaluation.

### Legality head
Implement a legality prediction head.
A practical first version may predict:
- legal source squares
- legal destination squares
or
- a dense [64, 64] legality matrix

Document the chosen design and keep it testable.

### Value head
Implement a scalar value head:
- output shape [B] or [B, 1]

### Concept head
Implement a multi-label concept head with a small default concept vocabulary.
A reasonable starter vocabulary includes tags like:
- check
- capture
- recapture
- pin
- fork
- skewer
- discovered_attack
- discovered_check
- king_safety
- passed_pawn
- open_file
- promotion_threat

Keep the vocabulary configurable.

## Future extension stubs (must exist, not fully implemented)
Create placeholder modules/interfaces for:
- future_fusion_stub.py
- future_rationale_stub.py

These should define minimal interfaces only.
Do not implement actual LLM fusion yet.

## Training requirements
Implement a minimal, correct supervised training path for the spatial baseline.

Required losses:
- policy loss
- state probe loss
- legality loss
- value loss
- concept loss

Keep loss composition configurable.

Training code must:
- be device-aware
- support CUDA when available
- run a tiny overfit smoke test
- avoid distributed complexity

## Evaluation requirements

### State fidelity metrics
Implement metrics for at least:
- piece occupancy reconstruction accuracy
- side-to-move accuracy
- castling-right accuracy
- en-passant accuracy
- in-check accuracy
- legality prediction quality

### Move quality metrics
Implement at least:
- top-1 move accuracy
- top-k move accuracy over legal moves

These may be basic in this phase.

### Reporting
Implement structured report output:
- JSON
- CSV if convenient

Evaluation must be callable from a script entry point.

## Fixtures and smoke data
Do not depend on an external internet download for smoke tests.

Create a small local fixture set using python-chess or hardcoded FENs:
- starting position
- simple opening positions
- castling positions
- en-passant positions
- promotion positions
- check/checkmate-threat positions

The fixture set only needs to be large enough for:
- codec tests
- shape tests
- tiny overfit runs
- evaluation smoke tests

## Tests (mandatory)
Write and pass tests for:
1. square index <-> tensor coordinate round-trip
2. FEN round-trip
3. BoardState round-trip
4. move factorization round-trip
5. legal move agreement with python-chess
6. tensor codec invariants
7. board encoder output shapes
8. factorized policy head output shapes
9. state probe output shapes
10. tiny-batch overfit smoke run
11. evaluation smoke run

Do not claim completion if these fail.

## Configuration requirements
Use YAML configs for:
- model hyperparameters
- training defaults
- baseline variants
- evaluation defaults

Avoid overengineering.
A small config system is enough.

## Documentation requirements
Write:
- `docs/architecture.md`
- `docs/data_schema.md`
- `docs/experiment_plan.md`
- `docs/ablations.md`

These docs must explain:
- coordinate conventions
- tensor schema
- head responsibilities
- loss structure
- baseline scope
- future phase boundaries

## Practical simplifications allowed
These are acceptable in this phase:
- single-GPU only
- no Stockfish integration
- no external dataset builder beyond local fixtures
- no full concept annotation pipeline
- no rationale generation implementation
- no RL

## Forbidden shortcuts
Do not:
- silently change coordinate conventions mid-codebase
- leak legal moves into input planes
- mix training and evaluation logic in one monolithic file
- create notebook-only workflows
- hardcode CUDA assumptions into tests
- claim a "world model" implementation without explicit state-probe support

## Done criteria for this phase
This phase is complete only if:
1. the repo skeleton exists
2. docs are written
3. codecs are implemented and tested
4. the spatial baseline model runs
5. the tiny overfit smoke test passes
6. the evaluation smoke test passes
7. CPU fallback works for tests
8. CUDA-aware training path exists
9. future fusion/rationale stubs exist
10. no forbidden shortcuts were used

## Output protocol for every Codex task
At the end of every task, provide:
1. changed files
2. tests run
3. test results
4. remaining blockers
5. explicit statement of anything left as a stub

Never claim completion if tests fail.