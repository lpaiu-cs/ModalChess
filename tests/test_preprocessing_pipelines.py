import json
from pathlib import Path

import chess

from modalchess.data.eval_enrichment import EvalEnrichmentConfig, enrich_supervised_records_with_eval
from modalchess.data.pgn_pilot import PgnPilotBuildConfig, build_supervised_records_from_pgn, derive_pgn_game_id
from modalchess.data.preprocessing_common import (
    StableSplitConfig,
    assert_history_fens_contract,
    assign_split_by_game_id,
    normalize_fen_for_eval_join,
)
from modalchess.data.puzzle_sidecar import transform_puzzle_row


def _write_multi_game_pgn(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                '[Event "Rated Blitz game"]',
                '[Site "https://lichess.org/game-a"]',
                '[Date "2024.01.01"]',
                '[Round "-"]',
                '[White "WhiteA"]',
                '[Black "BlackA"]',
                '[Result "*"]',
                '[Split "train"]',
                "",
                "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 *",
                "",
                '[Event "Rated Blitz game"]',
                '[Site "https://lichess.org/game-b"]',
                '[Date "2024.01.02"]',
                '[Round "-"]',
                '[White "WhiteB"]',
                '[Black "BlackB"]',
                '[Result "*"]',
                '[Split "val"]',
                "",
                "1. d4 d5 2. c4 e6 3. Nc3 Nf6 *",
                "",
                '[Event "Rated Blitz game"]',
                '[Site "https://lichess.org/game-c"]',
                '[Date "2024.01.03"]',
                '[Round "-"]',
                '[White "WhiteC"]',
                '[Black "BlackC"]',
                '[Result "*"]',
                '[Split "test"]',
                "",
                "1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 *",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_puzzle_transform_applies_first_move_before_target_extraction() -> None:
    row = {
        "PuzzleId": "p1",
        "GameId": "g1",
        "FEN": chess.STARTING_FEN,
        "Moves": "e2e4 e7e5",
        "Themes": "opening",
    }

    record = transform_puzzle_row(row)

    board = chess.Board(chess.STARTING_FEN)
    board.push_uci("e2e4")
    assert record is not None
    assert record["fen"] == board.fen(en_passant="fen")
    assert record["target_move_uci"] == "e7e5"

    board.push_uci("e7e5")
    assert record["next_fen"] == board.fen(en_passant="fen")


def test_eval_join_uses_first_four_fen_fields(tmp_path: Path) -> None:
    supervised_path = tmp_path / "supervised.jsonl"
    eval_path = tmp_path / "evals.jsonl"
    supervised_row = {
        "position_id": "p1",
        "game_id": "g1",
        "fen": "8/8/8/8/8/8/8/K6k w - - 7 32",
    }
    eval_row = {
        "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
        "cp": 42,
        "pv": "a1b1",
    }
    supervised_path.write_text(json.dumps(supervised_row) + "\n", encoding="utf-8")
    eval_path.write_text(json.dumps(eval_row) + "\n", encoding="utf-8")

    records, _ = enrich_supervised_records_with_eval(
        [supervised_path],
        [eval_path],
        EvalEnrichmentConfig(),
    )

    assert normalize_fen_for_eval_join(supervised_row["fen"]) == normalize_fen_for_eval_join(eval_row["fen"])
    assert records[0]["engine_eval_cp"] == 42
    assert records[0]["engine_best_move_uci"] == "a1b1"


def test_pgn_replay_produces_consistent_target_and_next_fen(tmp_path: Path) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(
        "\n".join(
            [
                '[Event "Rated Blitz game"]',
                '[Site "https://lichess.org/abc123"]',
                '[Date "2024.01.01"]',
                '[Round "-"]',
                '[White "White"]',
                '[Black "Black"]',
                '[Result "1-0"]',
                '[Split "train"]',
                "",
                "1. e4 e5 2. Nf3 Nc6 1-0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    records_by_split, _ = build_supervised_records_from_pgn([pgn_path], PgnPilotBuildConfig())
    records = records_by_split["train"]

    assert records[0]["fen"] == chess.Board().fen(en_passant="fen")
    assert records[0]["target_move_uci"] == "e2e4"

    board = chess.Board()
    board.push_uci("e2e4")
    assert records[0]["next_fen"] == board.fen(en_passant="fen")


def test_pgn_history_fens_contract_holds(tmp_path: Path) -> None:
    pgn_path = tmp_path / "history_sample.pgn"
    pgn_path.write_text(
        "\n".join(
            [
                '[Event "Rated Blitz game"]',
                '[Site "https://lichess.org/history"]',
                '[Date "2024.01.01"]',
                '[Round "-"]',
                '[White "White"]',
                '[Black "Black"]',
                '[Result "1-0"]',
                '[Split "train"]',
                "",
                "1. e4 e5 1-0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    records_by_split, _ = build_supervised_records_from_pgn([pgn_path], PgnPilotBuildConfig())
    second_record = records_by_split["train"][1]

    assert_history_fens_contract(
        second_record["position_id"],
        second_record["fen"],
        second_record["history_fens"],
    )
    assert second_record["history_fens"][-1] == second_record["fen"]


def test_game_id_split_is_stable_and_game_level(tmp_path: Path) -> None:
    pgn_path = tmp_path / "split_sample.pgn"
    pgn_path.write_text(
        "\n".join(
            [
                '[Event "Rated Blitz game"]',
                '[Site "https://lichess.org/stable"]',
                '[Date "2024.01.01"]',
                '[Round "-"]',
                '[White "White"]',
                '[Black "Black"]',
                '[Result "1-0"]',
                "",
                "1. e4 e5 2. Nf3 Nc6 1-0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = PgnPilotBuildConfig(
        split_config=StableSplitConfig(train_ratio=0.5, val_ratio=0.25, salt="seeded"),
    )
    records_by_split, _ = build_supervised_records_from_pgn([pgn_path], config)
    non_empty_splits = [name for name, records in records_by_split.items() if records]

    assert non_empty_splits
    assert len(non_empty_splits) == 1

    headers = {
        "Site": "https://lichess.org/stable",
        "Event": "Rated Blitz game",
        "Round": "-",
        "White": "White",
        "Black": "Black",
        "Date": "2024.01.01",
    }
    game_id = derive_pgn_game_id(headers)
    expected_split = assign_split_by_game_id(game_id, config.split_config)
    assert non_empty_splits[0] == expected_split


def test_pgn_builder_respects_max_games_limit(tmp_path: Path) -> None:
    pgn_path = tmp_path / "multi_games.pgn"
    _write_multi_game_pgn(pgn_path)

    records_by_split, report = build_supervised_records_from_pgn(
        [pgn_path],
        PgnPilotBuildConfig(max_games=1),
    )

    assert report["games_kept"] == 1
    assert report["stop_reason"] == "max_games_reached"
    assert sum(len(records) for records in records_by_split.values()) == 6


def test_pgn_builder_respects_max_positions_limit(tmp_path: Path) -> None:
    pgn_path = tmp_path / "limited_positions.pgn"
    _write_multi_game_pgn(pgn_path)

    records_by_split, report = build_supervised_records_from_pgn(
        [pgn_path],
        PgnPilotBuildConfig(max_positions=4),
    )

    assert report["positions_written"] == 4
    assert report["stop_reason"] == "max_positions_reached"
    assert sum(len(records) for records in records_by_split.values()) == 4


def test_pgn_builder_sample_every_n_plies_is_deterministic(tmp_path: Path) -> None:
    pgn_path = tmp_path / "deterministic_sampling.pgn"
    _write_multi_game_pgn(pgn_path)

    config = PgnPilotBuildConfig(sample_every_n_plies=3, random_seed=17)
    first_records, _ = build_supervised_records_from_pgn([pgn_path], config)
    second_records, _ = build_supervised_records_from_pgn([pgn_path], config)

    def flatten_targets(records_by_split: dict[str, list[dict[str, object]]]) -> list[str]:
        flat: list[str] = []
        for split_name in ("train", "val", "test"):
            flat.extend(str(row["target_move_uci"]) for row in records_by_split[split_name])
        return flat

    assert flatten_targets(first_records) == flatten_targets(second_records)


def test_pgn_builder_respects_max_positions_per_game(tmp_path: Path) -> None:
    pgn_path = tmp_path / "per_game_cap.pgn"
    _write_multi_game_pgn(pgn_path)

    records_by_split, report = build_supervised_records_from_pgn(
        [pgn_path],
        PgnPilotBuildConfig(max_positions_per_game=2),
    )

    per_game_counts: dict[str, int] = {}
    for split_records in records_by_split.values():
        for row in split_records:
            per_game_counts[str(row["game_id"])] = per_game_counts.get(str(row["game_id"]), 0) + 1
    assert per_game_counts
    assert all(count <= 2 for count in per_game_counts.values())
    assert report["positions_skipped_per_game_cap"] > 0
