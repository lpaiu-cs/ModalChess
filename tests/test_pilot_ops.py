import json
from pathlib import Path

from modalchess.data.pilot_report import generate_pilot_data_report
from modalchess.data.mate_sidecar import transform_mate_row
from modalchess.data.source_fetch import FetchedArtifact, write_fetch_lock_manifest


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_write_fetch_lock_manifest_includes_notes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "raw_fetch_lock.yaml"
    notes_path = tmp_path / "raw_fetch_notes.md"
    entries = [
        FetchedArtifact(
            source_name="source_a",
            source_url="https://example.com/a",
            version="v1",
            local_path=str(tmp_path / "a.bin"),
            file_size=123,
            checksum="abc",
            fetch_timestamp="2026-03-19T00:00:00+00:00",
            status="downloaded",
            note=None,
        ),
        FetchedArtifact(
            source_name="source_b",
            source_url="https://example.com/b",
            version="v2",
            local_path=None,
            file_size=None,
            checksum=None,
            fetch_timestamp="2026-03-19T00:00:00+00:00",
            status="snapshot_only",
            note="manual step required",
        ),
    ]

    payload = write_fetch_lock_manifest(entries=entries, manifest_path=manifest_path, notes_path=notes_path)

    assert manifest_path.exists()
    assert notes_path.exists()
    assert len(payload["artifacts"]) == 2
    assert "manual step required" in notes_path.read_text(encoding="utf-8")


def test_generate_pilot_data_report_passes_for_valid_dataset(tmp_path: Path) -> None:
    root = tmp_path / "real_v1"
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True)
    (manifests_dir / "pgn_manifest.yaml").write_text(
        "report:\n  drop_reasons:\n    too_short_game: 3\n",
        encoding="utf-8",
    )

    train_rows = [
        {
            "position_id": "train_1",
            "game_id": "g_train",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "history_fens": ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
            "target_move_uci": "e2e4",
            "next_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "engine_eval_cp": 12,
            "concept_tags": ["king_safety"],
        }
    ]
    val_rows = [
        {
            "position_id": "val_1",
            "game_id": "g_val",
            "fen": "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            "history_fens": ["4k3/P7/8/8/8/8/8/4K3 w - - 0 1"],
            "target_move_uci": "a7a8q",
            "next_fen": "Q3k3/8/8/8/8/8/8/4K3 b - - 0 1",
        }
    ]
    test_rows = [
        {
            "position_id": "test_1",
            "game_id": "g_test",
            "fen": "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
            "history_fens": ["4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"],
            "target_move_uci": "e1e2",
            "next_fen": "4k3/8/8/8/8/8/4K3/8 b - - 0 1",
        }
    ]
    _write_jsonl(root / "supervised_train.jsonl", train_rows)
    _write_jsonl(root / "supervised_val.jsonl", val_rows)
    _write_jsonl(root / "supervised_test.jsonl", test_rows)

    report = generate_pilot_data_report(input_root=root, min_val_rows=1, min_test_rows=1)

    assert report["passes_gate"] is True
    assert report["splits"]["train"]["row_count"] == 1
    assert report["splits"]["val"]["subset_counts"]["promotion"] == 1
    assert report["dropped_row_counts_by_reason"]["too_short_game"] == 3


def test_generate_pilot_data_report_detects_split_leakage(tmp_path: Path) -> None:
    root = tmp_path / "real_v1"
    _write_jsonl(
        root / "supervised_train.jsonl",
        [
            {
                "position_id": "train_1",
                "game_id": "shared_game",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "history_fens": ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
                "target_move_uci": "e2e4",
                "next_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            }
        ],
    )
    _write_jsonl(
        root / "supervised_val.jsonl",
        [
            {
                "position_id": "val_1",
                "game_id": "val_game",
                "fen": "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                "history_fens": ["4k3/P7/8/8/8/8/8/4K3 w - - 0 1"],
                "target_move_uci": "a7a8q",
                "next_fen": "Q3k3/8/8/8/8/8/8/4K3 b - - 0 1",
            }
        ],
    )
    _write_jsonl(
        root / "supervised_test.jsonl",
        [
            {
                "position_id": "test_1",
                "game_id": "shared_game",
                "fen": "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
                "history_fens": ["4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"],
                "target_move_uci": "e1e2",
                "next_fen": "4k3/8/8/8/8/8/4K3/8 b - - 0 1",
            }
        ],
    )

    report = generate_pilot_data_report(input_root=root, min_val_rows=1, min_test_rows=1)

    assert report["passes_gate"] is False
    assert report["gate_checks"]["no_split_leakage_by_game_id"] is False


def test_transform_mate_row_parses_prompt_style_input() -> None:
    row = {
        "input": (
            'The FEN of the given chess board is '
            '"8/6kp/2Nr2p1/3n1p2/7P/3R2P1/5PK1/8 w - - 0 42". '
            "Which move is better? MoveA:c6e5, Improve activity. "
            "TacticA: c6e5 g7f6 MoveB:c6d8, Retreat. TacticB: c6d8 d5f4"
        ),
        "output": "MoveA:c6e5",
    }

    record = transform_mate_row(row)

    assert record["fen"] == "8/6kp/2Nr2p1/3n1p2/7P/3R2P1/5PK1/8 w - - 0 42"
    assert record["candidate_moves"] == ["c6e5", "c6d8"]
    assert record["strategy_text"] is not None
    assert record["tactic_text"] is not None
    assert record["preferred_move"] == "MoveA:c6e5"
