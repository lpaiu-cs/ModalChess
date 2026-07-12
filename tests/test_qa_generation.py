"""P0 QA 생성기/검증기 테스트: 알려진 정답, 생성-검증 왕복, 균형, 거짓 전제."""

from __future__ import annotations

import random

import chess

from modalchess.fusion.qa_generator import generate_split
from modalchess.fusion.qa_tasks import HELD_OUT_TEMPLATE_INDEX, TEMPLATES, no_such_piece
from modalchess.fusion.qa_verifier import verify_corpus, verify_item

START = chess.STARTING_FEN
# 흑 나이트 e5가 Re1에 의해 e8 킹에 절대 핀 (e파일 개방)
PIN_FEN = "4k3/8/8/4n3/8/8/8/4R2K b - - 0 1"
# 백이 Qh4+에 체크당한 포지션
CHECK_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"


def _item(task, fen, params, answer, candidates, template_id=0):
    from modalchess.fusion.qa_tasks import TASK_TIER

    return {
        "qa_id": "t",
        "tier": TASK_TIER[task],
        "task": task,
        "template_id": template_id,
        "question": TEMPLATES[task][template_id].format(**params),
        "answer": answer,
        "candidates": candidates,
        "params": params,
        "fen": fen,
    }


def test_verifier_known_answers_start_position() -> None:
    from modalchess.fusion.qa_tasks import PIECE_ON_SQUARE_CANDIDATES

    good = [
        _item("piece_on_square", START, {"square": "e1"}, "white king",
              list(PIECE_ON_SQUARE_CANDIDATES)),
        _item("piece_on_square", START, {"square": "d5"}, "empty",
              list(PIECE_ON_SQUARE_CANDIDATES)),
        _item("side_to_move", START, {}, "white", ["white", "black"]),
        _item("castling_right", START, {"color": "black", "side": "queenside"}, "yes",
              ["yes", "no"]),
        _item("piece_count", START, {"color": "white", "piece": "knight"}, "2 or more",
              ["0", "1", "2 or more"]),
        _item("square_attacked", START, {"square": "f3", "color": "white"}, "yes",
              ["yes", "no"]),
        _item("is_check", START, {}, "no", ["yes", "no"]),
        _item("piece_defended", START, {"square": "e4", "color": "white"},
              no_such_piece("e4", "white"), ["yes", "no", no_such_piece("e4", "white")]),
        _item("piece_pinned", START, {"square": "e2"}, "no",
              ["yes", "no", no_such_piece("e2")]),
    ]
    for item in good:
        assert verify_item(item) == [], (item["task"], verify_item(item))


def test_verifier_catches_wrong_answer() -> None:
    from modalchess.fusion.qa_tasks import PIECE_ON_SQUARE_CANDIDATES

    bad = _item("piece_on_square", START, {"square": "e1"}, "white queen",
                list(PIECE_ON_SQUARE_CANDIDATES))
    errs = verify_item(bad)
    assert any("answer mismatch" in e for e in errs)


def test_verifier_pin_and_check_fixtures() -> None:
    board = chess.Board(PIN_FEN)
    assert board.is_pinned(chess.BLACK, chess.E5)  # fixture 자체 검증
    pinned = _item("piece_pinned", PIN_FEN, {"square": "e5"}, "yes",
                   ["yes", "no", no_such_piece("e5")])
    assert verify_item(pinned) == []

    assert chess.Board(CHECK_FEN).is_check()
    check = _item("is_check", CHECK_FEN, {}, "yes", ["yes", "no"])
    assert verify_item(check) == []


def test_verifier_t3_known_answers() -> None:
    # 1.e4 후 흑 차례
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    # move_is_legal: d7d5 합법, d7d4(3칸) 비합법
    assert verify_item(_item("move_is_legal", after_e4, {"frm": "d7", "to": "d5"}, "yes",
                             ["yes", "no"])) == []
    assert verify_item(_item("move_is_legal", after_e4, {"frm": "d7", "to": "d4"}, "no",
                             ["yes", "no"])) == []
    # move_is_capture: 스카치 캡처 위치 — 백 e4 폰이 d5를 따먹음
    cap = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    assert chess.Board(cap).is_capture(chess.Move.from_uci("e4d5"))
    assert verify_item(_item("move_is_capture", cap, {"frm": "e4", "to": "d5"}, "yes",
                             ["yes", "no"])) == []
    assert verify_item(_item("move_is_capture", cap, {"frm": "g1", "to": "f3"}, "no",
                             ["yes", "no"])) == []
    # move_gives_check: Qh5 후 흑 ...g6, Qxg6 아니고 — 간단히 Qh5xf7는 체크. 직접 구성:
    chk = "rnbqkbnr/pppp1ppp/8/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 1"  # 백 퀸 h5
    assert chess.Board(chk).gives_check(chess.Move.from_uci("h5f7"))
    assert verify_item(_item("move_gives_check", chk, {"frm": "h5", "to": "f7"}, "yes",
                             ["yes", "no"])) == []
    assert verify_item(_item("move_gives_check", chk, {"frm": "g1", "to": "f3"}, "no",
                             ["yes", "no"])) == []


def _random_position_pool(n_games: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pool: list[tuple[str, str]] = [(START, "g0"), (PIN_FEN, "g_pin"), (CHECK_FEN, "g_chk")]
    for g in range(n_games):
        board = chess.Board()
        for _ in range(rng.randrange(10, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            pool.append((board.fen(), f"g{g + 1}"))
    return pool


def test_generator_roundtrip_all_items_verified() -> None:
    pool = _random_position_pool(n_games=40, seed=7)
    items, _shortfall = generate_split(
        pool, quota_scale=0.002, seed=11, include_held_out_template=False,
    )
    assert len(items) > 50
    report = verify_corpus(items)
    assert report["n_mismatched"] == 0, report["error_samples"]
    assert all(item["template_id"] != HELD_OUT_TEMPLATE_INDEX for item in items)


def test_generator_balance_and_false_premise() -> None:
    pool = _random_position_pool(n_games=60, seed=3)
    items, _ = generate_split(pool, quota_scale=0.003, seed=17, include_held_out_template=True)
    by_task_class: dict[tuple[str, str], int] = {}
    for item in items:
        answer = str(item["answer"])
        cls = "nosuch" if answer.startswith("there is no") else answer
        key = (item["task"], cls)
        by_task_class[key] = by_task_class.get(key, 0) + 1
    # 쿼터 균형: side_to_move는 흔하므로 두 클래스가 정확히 쿼터만큼 채워진다
    quota = max(1, round(6500 * 0.003))
    assert by_task_class[("side_to_move", "white")] == quota
    assert by_task_class[("side_to_move", "black")] == quota
    # 거짓 전제 항목이 실제로 생성된다
    assert by_task_class.get(("piece_defended", "nosuch"), 0) >= 1
    assert by_task_class.get(("piece_pinned", "nosuch"), 0) >= 1


def test_tiers_filter_controls_t3() -> None:
    from modalchess.fusion.qa_tasks import TASK_TIER

    pool = _random_position_pool(n_games=40, seed=9)
    # 기본(T1,T2) = qa_v1: T3 task가 없어야 한다 (재현성 — T3 누출 방지)
    items_v1, _ = generate_split(pool, quota_scale=0.002, seed=11,
                                 include_held_out_template=False)
    assert all(TASK_TIER[i["task"]] in ("T1", "T2") for i in items_v1)
    assert not any(TASK_TIER[i["task"]] == "T3" for i in items_v1)
    # T1,T2,T3 = qa_v2: T3 task가 실제로 생성돼야 한다
    items_v2, _ = generate_split(pool, quota_scale=0.002, seed=11,
                                 include_held_out_template=False, tiers=("T1", "T2", "T3"))
    assert any(TASK_TIER[i["task"]] == "T3" for i in items_v2)


def test_tiers_validation_rejects_unknown_and_empty() -> None:
    import pytest

    pool = _random_position_pool(n_games=5, seed=2)
    for bad in [("T1", "T22"), (), ("",)]:
        with pytest.raises(ValueError):
            generate_split(pool, quota_scale=0.001, seed=11,
                           include_held_out_template=False, tiers=bad)


def test_generator_no_duplicate_params_per_position() -> None:
    pool = _random_position_pool(n_games=20, seed=5)
    items, _ = generate_split(pool, quota_scale=0.002, seed=23, include_held_out_template=False)
    sigs = [(i["task"], i["position_key"], repr(sorted(i["params"].items()))) for i in items]
    assert len(sigs) == len(set(sigs))
