"""grounded QA 생성기: 클래스 쿼터 균형 + 표적 샘플링 (docs/phase2_plan.md §4-5).

정답 계산은 여기의 API 경로(piece_at / is_attacked_by / is_pinned / has_*_castling_rights /
len(board.pieces()) / board.is_check)로 하고, qa_verifier는 의도적으로 다른 경로
(piece_map / attackers / pin mask / castling_rights 비트마스크 / pieces_mask popcount /
checkers)로 재계산한다. 불일치 1건 = P0 실패.
"""

from __future__ import annotations

import random
from typing import Any, Callable

import chess

from modalchess.fusion.qa_tasks import (
    ALL_TASKS,
    CASTLE_SIDES,
    COLOR_NAMES,
    COUNT_CANDIDATES,
    COUNT_PIECE_TYPES,
    EMPTY,
    HELD_OUT_TEMPLATE_INDEX,
    KING_SQUARE_NUM_CANDIDATES,
    NO,
    PIECE_ON_SQUARE_CANDIDATES,
    PIECE_WORDS,
    SIDE_CANDIDATES,
    TASK_TIER,
    TEMPLATES,
    YES,
    no_such_piece,
    piece_label,
    position_key,
)

_MAX_ITEMS_PER_POSITION = 3


def default_quotas(scale: float) -> dict[str, dict[Any, int]]:
    """과제별 클래스 쿼터 (train 기준량 × scale)."""

    def q(n: int) -> int:
        return max(1, round(n * scale))

    quotas: dict[str, dict[Any, int]] = {
        "piece_on_square": {c: q(1000) for c in PIECE_ON_SQUARE_CANDIDATES},
        "king_square": {c: q(6500) for c in ("white", "black")},
        "side_to_move": {c: q(6500) for c in SIDE_CANDIDATES},
        "castling_right": {
            (color, side, ans): q(1625)
            for color in ("white", "black")
            for side in CASTLE_SIDES
            for ans in (YES, NO)
        },
        "piece_count": {c: q(4333) for c in COUNT_CANDIDATES},
        "square_attacked": {c: q(6500) for c in (YES, NO)},
        "piece_defended": {c: q(4333) for c in (YES, NO, "nosuch")},
        "is_check": {c: q(6500) for c in (YES, NO)},
        "piece_pinned": {c: q(4333) for c in (YES, NO, "nosuch")},
        "move_is_capture": {c: q(6500) for c in (YES, NO)},
        "move_gives_check": {c: q(6500) for c in (YES, NO)},
        "move_is_legal": {c: q(6500) for c in (YES, NO)},
    }
    return quotas


def _count_bucket(n: int) -> str:
    if n >= 2:
        return "2 or more"
    return str(n)


def _needed(remaining: dict[Any, int], rng: random.Random) -> list[Any]:
    keys = [k for k, v in remaining.items() if v > 0]
    rng.shuffle(keys)
    return keys


class _SplitGenerator:
    def __init__(self, quotas: dict[str, dict[Any, int]], template_ids: list[int], rng: random.Random) -> None:
        self.remaining = {task: dict(classes) for task, classes in quotas.items()}
        self.template_ids = template_ids
        self.rng = rng
        self.seen: set[tuple[str, str, str]] = set()
        self.counter = 0
        self.builders: dict[str, Callable[[chess.Board], dict | None]] = {
            "piece_on_square": self._piece_on_square,
            "king_square": self._king_square,
            "side_to_move": self._side_to_move,
            "castling_right": self._castling_right,
            "piece_count": self._piece_count,
            "square_attacked": self._square_attacked,
            "piece_defended": self._piece_defended,
            "is_check": self._is_check,
            "piece_pinned": self._piece_pinned,
            "move_is_capture": self._move_is_capture,
            "move_gives_check": self._move_gives_check,
            "move_is_legal": self._move_is_legal,
        }

    # --- item 공통 조립 ---

    def _make(self, task: str, board: chess.Board, cls: Any, answer: str,
              candidates: list[str], params: dict[str, Any], fmt: dict[str, str]) -> dict | None:
        sig = (task, position_key(board.fen()), repr(sorted(params.items())))
        if sig in self.seen:
            return None
        self.seen.add(sig)
        template_id = self.rng.choice(self.template_ids)
        question = TEMPLATES[task][template_id].format(**fmt)
        assert answer in candidates and len(set(candidates)) == len(candidates)
        self.remaining[task][cls] -= 1
        self.counter += 1
        return {
            "qa_id": f"{task}_{self.counter:07d}",
            "tier": TASK_TIER[task],
            "task": task,
            "template_id": template_id,
            "question": question,
            "answer": answer,
            "candidates": candidates,
            "params": params,
        }

    # --- 과제별 빌더: 이 포지션에서 미충족 클래스를 채울 수 있으면 item 반환 ---

    def _piece_on_square(self, board: chess.Board) -> dict | None:
        for cls in _needed(self.remaining["piece_on_square"], self.rng):
            if cls == EMPTY:
                squares = [s for s in chess.SQUARES if board.piece_at(s) is None]
            else:
                color_name, piece_word = cls.split()
                color = color_name == "white"
                ptype = {v: k for k, v in PIECE_WORDS.items()}[piece_word]
                squares = list(board.pieces(ptype, color))
            if not squares:
                continue
            sq = self.rng.choice(squares)
            sq_name = chess.square_name(sq)
            return self._make(
                "piece_on_square", board, cls, cls, list(PIECE_ON_SQUARE_CANDIDATES),
                {"square": sq_name}, {"square": sq_name},
            )
        return None

    def _king_square(self, board: chess.Board) -> dict | None:
        for cls in _needed(self.remaining["king_square"], self.rng):
            color = cls == "white"
            king_sq = board.king(color)
            if king_sq is None:
                continue
            answer = chess.square_name(king_sq)
            pool: list[int] = []
            neighbors = [s for s in chess.SQUARES if chess.square_distance(s, king_sq) == 1]
            self.rng.shuffle(neighbors)
            pool.extend(neighbors[:2])
            same_file = [s for s in chess.SQUARES
                         if chess.square_file(s) == chess.square_file(king_sq) and s != king_sq]
            same_rank = [s for s in chess.SQUARES
                         if chess.square_rank(s) == chess.square_rank(king_sq) and s != king_sq]
            pool.append(self.rng.choice(same_file))
            pool.append(self.rng.choice(same_rank))
            candidates = {answer}
            for s in pool:
                candidates.add(chess.square_name(s))
            while len(candidates) < KING_SQUARE_NUM_CANDIDATES:
                candidates.add(chess.square_name(self.rng.randrange(64)))
            cand_list = sorted(candidates)
            self.rng.shuffle(cand_list)
            return self._make(
                "king_square", board, cls, answer, cand_list[:],
                {"color": cls}, {"color": cls},
            )
        return None

    def _side_to_move(self, board: chess.Board) -> dict | None:
        cls = COLOR_NAMES[board.turn]
        if self.remaining["side_to_move"].get(cls, 0) <= 0:
            return None
        return self._make("side_to_move", board, cls, cls, list(SIDE_CANDIDATES), {}, {})

    def _castling_right(self, board: chess.Board) -> dict | None:
        combos = list(self.remaining["castling_right"].keys())
        self.rng.shuffle(combos)
        for color_name, side, needed_ans in combos:
            if self.remaining["castling_right"][(color_name, side, needed_ans)] <= 0:
                continue
            color = color_name == "white"
            if side == "kingside":
                ans = YES if board.has_kingside_castling_rights(color) else NO
            else:
                ans = YES if board.has_queenside_castling_rights(color) else NO
            if ans != needed_ans:
                continue
            return self._make(
                "castling_right", board, (color_name, side, ans), ans, [YES, NO],
                {"color": color_name, "side": side}, {"color": color_name, "side": side},
            )
        return None

    def _piece_count(self, board: chess.Board) -> dict | None:
        combos = [(c, p) for c in (chess.WHITE, chess.BLACK) for p in COUNT_PIECE_TYPES]
        self.rng.shuffle(combos)
        for color, ptype in combos:
            bucket = _count_bucket(len(board.pieces(ptype, color)))
            if self.remaining["piece_count"].get(bucket, 0) <= 0:
                continue
            color_name, piece_word = COLOR_NAMES[color], PIECE_WORDS[ptype]
            return self._make(
                "piece_count", board, bucket, bucket, list(COUNT_CANDIDATES),
                {"color": color_name, "piece": piece_word},
                {"color": color_name, "piece": piece_word},
            )
        return None

    def _square_attacked(self, board: chess.Board) -> dict | None:
        for _ in range(8):
            color = self.rng.choice((chess.WHITE, chess.BLACK))
            sq = self.rng.randrange(64)
            ans = YES if board.is_attacked_by(color, sq) else NO
            if self.remaining["square_attacked"].get(ans, 0) <= 0:
                continue
            sq_name, color_name = chess.square_name(sq), COLOR_NAMES[color]
            return self._make(
                "square_attacked", board, ans, ans, [YES, NO],
                {"square": sq_name, "color": color_name},
                {"square": sq_name, "color": color_name},
            )
        return None

    def _piece_defended(self, board: chess.Board) -> dict | None:
        for cls in _needed(self.remaining["piece_defended"], self.rng):
            if cls == "nosuch":
                color = self.rng.choice((chess.WHITE, chess.BLACK))
                squares = [s for s in chess.SQUARES
                           if (p := board.piece_at(s)) is None or p.color != color]
                if not squares:
                    continue
                sq = self.rng.choice(squares)
                sq_name, color_name = chess.square_name(sq), COLOR_NAMES[color]
                answer = no_such_piece(sq_name, color_name)
            else:
                found = None
                for color in (chess.WHITE, chess.BLACK):
                    squares = [s for s in chess.SQUARES
                               if (p := board.piece_at(s)) is not None and p.color == color]
                    self.rng.shuffle(squares)
                    for s in squares:
                        ans = YES if board.is_attacked_by(color, s) else NO
                        if ans == cls:
                            found = (color, s)
                            break
                    if found:
                        break
                if not found:
                    continue
                color, sq = found
                sq_name, color_name = chess.square_name(sq), COLOR_NAMES[color]
                answer = cls
            candidates = [YES, NO, no_such_piece(sq_name, color_name)]
            return self._make(
                "piece_defended", board, cls, answer, candidates,
                {"square": sq_name, "color": color_name},
                {"square": sq_name, "color": color_name},
            )
        return None

    def _is_check(self, board: chess.Board) -> dict | None:
        ans = YES if board.is_check() else NO
        if self.remaining["is_check"].get(ans, 0) <= 0:
            return None
        return self._make("is_check", board, ans, ans, [YES, NO], {}, {})

    def _piece_pinned(self, board: chess.Board) -> dict | None:
        for cls in _needed(self.remaining["piece_pinned"], self.rng):
            if cls == "nosuch":
                squares = [s for s in chess.SQUARES if board.piece_at(s) is None]
                if not squares:
                    continue
                sq = self.rng.choice(squares)
                sq_name = chess.square_name(sq)
                answer = no_such_piece(sq_name)
            else:
                found = None
                squares = [s for s in chess.SQUARES if board.piece_at(s) is not None]
                self.rng.shuffle(squares)
                for s in squares:
                    piece = board.piece_at(s)
                    ans = YES if board.is_pinned(piece.color, s) else NO
                    if ans == cls:
                        found = s
                        break
                if found is None:
                    continue
                sq = found
                sq_name = chess.square_name(sq)
                answer = cls
            candidates = [YES, NO, no_such_piece(sq_name)]
            return self._make(
                "piece_pinned", board, cls, answer, candidates,
                {"square": sq_name}, {"square": sq_name},
            )
        return None

    # --- T3: 1수 동역학 (move는 non-promotion만, {frm}→{to} 명시) ---

    def _legal_nonpromo(self, board: chess.Board) -> list[chess.Move]:
        return [m for m in board.legal_moves if m.promotion is None]

    def _move_item(self, task: str, board: chess.Board, cls: str, move: chess.Move) -> dict | None:
        frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
        return self._make(
            task, board, cls, cls, [YES, NO],
            {"frm": frm, "to": to}, {"frm": frm, "to": to},
        )

    def _move_is_capture(self, board: chess.Board) -> dict | None:
        for cls in _needed(self.remaining["move_is_capture"], self.rng):
            moves = [m for m in self._legal_nonpromo(board)
                     if (YES if board.is_capture(m) else NO) == cls]
            if not moves:
                continue
            return self._move_item("move_is_capture", board, cls, self.rng.choice(moves))
        return None

    def _move_gives_check(self, board: chess.Board) -> dict | None:
        for cls in _needed(self.remaining["move_gives_check"], self.rng):
            moves = [m for m in self._legal_nonpromo(board)
                     if (YES if board.gives_check(m) else NO) == cls]
            if not moves:
                continue
            return self._move_item("move_gives_check", board, cls, self.rng.choice(moves))
        return None

    def _move_is_legal(self, board: chess.Board) -> dict | None:
        legal = set(self._legal_nonpromo(board))
        for cls in _needed(self.remaining["move_is_legal"], self.rng):
            if cls == YES:
                if not legal:
                    continue
                return self._move_item("move_is_legal", board, YES, self.rng.choice(list(legal)))
            # NO: 착수측 기물이 있는 칸에서 비합법 도착으로 그럴듯한 illegal move 구성
            own_squares = [s for s in chess.SQUARES
                           if (p := board.piece_at(s)) is not None and p.color == board.turn]
            self.rng.shuffle(own_squares)
            for frm in own_squares:
                targets = list(range(64))
                self.rng.shuffle(targets)
                for to in targets:
                    if to == frm:
                        continue
                    mv = chess.Move(frm, to)
                    if mv not in legal and chess.Move(frm, to, promotion=chess.QUEEN) not in board.legal_moves:
                        return self._move_item("move_is_legal", board, NO, mv)
        return None

    # --- 메인 루프 ---

    def total_remaining(self) -> int:
        return sum(v for classes in self.remaining.values() for v in classes.values() if v > 0)

    def consume(self, fen: str, game_id: str) -> list[dict]:
        try:
            board = chess.Board(fen)
        except ValueError:
            return []
        items: list[dict] = []
        # self.remaining에 있는 task만 순회(tier 필터로 일부 task는 quota에서 제외될 수 있음).
        tasks = [t for t in self.remaining if any(v > 0 for v in self.remaining[t].values())]
        self.rng.shuffle(tasks)
        for task in tasks:
            if len(items) >= _MAX_ITEMS_PER_POSITION:
                break
            item = self.builders[task](board)
            if item is not None:
                item["fen"] = fen
                item["game_id"] = game_id
                item["position_key"] = position_key(fen)
                items.append(item)
        return items


def generate_split(
    positions: list[tuple[str, str]],
    quota_scale: float,
    seed: int,
    include_held_out_template: bool,
    tiers: tuple[str, ...] = ("T1", "T2"),
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """포지션 풀에서 클래스 균형 QA를 생성. 반환: (items, 미충족 쿼터 리포트).

    tiers: 생성할 tier 집합. 기본 ("T1","T2")=qa_v1 재현. ("T1","T2","T3")=qa_v2.
    코퍼스 버전별 task-set을 명시해 재현성 보장(T3가 qa_v1에 새는 것 방지).
    """
    from modalchess.fusion.qa_tasks import TASK_TIER

    rng = random.Random(seed)
    n_templates = len(next(iter(TEMPLATES.values())))
    template_ids = list(range(n_templates))
    if not include_held_out_template:
        template_ids.remove(HELD_OUT_TEMPLATE_INDEX)
    allowed = {t for t in ALL_TASKS if TASK_TIER[t] in tiers}
    quotas = {t: q for t, q in default_quotas(quota_scale).items() if t in allowed}
    gen = _SplitGenerator(quotas, template_ids, rng)
    order = list(range(len(positions)))
    rng.shuffle(order)
    items: list[dict] = []
    for idx in order:
        if gen.total_remaining() <= 0:
            break
        fen, game_id = positions[idx]
        items.extend(gen.consume(fen, game_id))
    shortfall = {
        task: {str(k): v for k, v in classes.items() if v > 0}
        for task, classes in gen.remaining.items()
        if any(v > 0 for v in classes.values())
    }
    return items, shortfall
