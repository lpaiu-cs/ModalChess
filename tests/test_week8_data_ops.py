from __future__ import annotations

import json
from pathlib import Path

import chess
import torch
import yaml

from modalchess.data.aux_language import build_aux_language_corpora
from modalchess.data.aux_source_fetch import record_local_aux_source, write_aux_fetch_lock
from modalchess.data.preprocessing_common import StableSplitConfig
from modalchess.data.probe_corpora import ProbeCorpusConfig, build_probe_corpora
from modalchess.data.probe_reports import write_probe_reports
from modalchess.data.target_realism import write_target_realism_report
from modalchess.eval.raw_text_retrieval import run_raw_text_retrieval_probes


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_all_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _embedding_payload(
    probe_ids: list[str],
    split_name: str,
    features: list[list[float]],
) -> dict[str, object]:
    return {
        "probe_id": probe_ids,
        "position_id": probe_ids,
        "source": ["synthetic"] * len(features),
        "split": [split_name] * len(features),
        "fen": [chess.STARTING_FEN] * len(features),
        "target_move_uci": [None] * len(features),
        "board_pooled": torch.tensor(features, dtype=torch.float32),
        "context_pooled": torch.tensor(features, dtype=torch.float32),
        "checkpoint_path": "dummy",
        "seed": 11,
        "git_hash": "dummy",
        "model_parameter_count": 1,
    }


def test_probe_corpora_grouped_split_and_manifest(tmp_path: Path) -> None:
    mate_path = tmp_path / "mate.jsonl"
    puzzle_path = tmp_path / "puzzle.jsonl"
    _write_jsonl(
        mate_path,
        [
            {
                "source_row_id": "mate_1",
                "fen": chess.STARTING_FEN,
                "strategy_text": "Develop pieces.",
                "tactic_text": "Keep the king safe.",
                "metadata": {"game_id": "shared_game"},
            },
            {
                "source_row_id": "mate_2",
                "fen": chess.STARTING_FEN,
                "strategy_text": "Control the center.",
                "tactic_text": "Look for tactics.",
                "metadata": {"game_id": "shared_game"},
            },
        ],
    )
    _write_jsonl(
        puzzle_path,
        [
            {
                "source_row_id": "puzzle_1",
                "fen": chess.STARTING_FEN,
                "target_move_uci": "e2e4",
                "theme_tags": ["fork"],
            }
        ],
    )
    output_root = tmp_path / "probe"
    result = build_probe_corpora(
        mate_path=mate_path,
        puzzle_path=puzzle_path,
        output_root=output_root,
        config=ProbeCorpusConfig(
            split_config=StableSplitConfig(salt="week8_test"),
            prefer_game_id_group_split=True,
            min_game_id_group_size=2,
        ),
    )
    strategy = result["split_strategy_by_source"]["mate"]
    assert strategy["split_key_type"] == "game_id"
    assert strategy["repeated_group_count"] == 1
    assert strategy["max_group_size"] == 2
    rows = []
    for split_name in ("train", "val", "test"):
        rows.extend(_read_all_rows(output_root / f"mate_{split_name}.jsonl"))
    shared_splits = {str(row["split"]) for row in rows if row["metadata"]["game_id"] == "shared_game"}
    assert len(shared_splits) == 1

    write_probe_reports(input_root=output_root, output_dir=output_root / "reports")
    report = json.loads((output_root / "reports" / "probe_corpora_report.json").read_text(encoding="utf-8"))
    assert report["split_strategy_by_source"]["mate"]["split_key_type"] == "game_id"


def test_probe_corpora_source_row_fallback_manifest(tmp_path: Path) -> None:
    mate_path = tmp_path / "mate.jsonl"
    puzzle_path = tmp_path / "puzzle.jsonl"
    _write_jsonl(
        mate_path,
        [
            {
                "source_row_id": "mate_1",
                "fen": chess.STARTING_FEN,
                "strategy_text": "Develop pieces.",
                "tactic_text": "Keep the king safe.",
                "metadata": {"game_id": "unique_1"},
            },
            {
                "source_row_id": "mate_2",
                "fen": chess.STARTING_FEN,
                "strategy_text": "Control the center.",
                "tactic_text": "Look for tactics.",
                "metadata": {"game_id": "unique_2"},
            },
        ],
    )
    _write_jsonl(
        puzzle_path,
        [
            {
                "source_row_id": "puzzle_1",
                "fen": chess.STARTING_FEN,
                "target_move_uci": "e2e4",
                "theme_tags": ["fork"],
            }
        ],
    )
    result = build_probe_corpora(
        mate_path=mate_path,
        puzzle_path=puzzle_path,
        output_root=tmp_path / "probe",
        config=ProbeCorpusConfig(
            split_config=StableSplitConfig(salt="week8_test"),
            prefer_game_id_group_split=True,
            min_game_id_group_size=2,
        ),
    )
    strategy = result["split_strategy_by_source"]["mate"]
    assert strategy["split_key_type"] == "source_row_id"
    assert strategy["fallback_reason"] == "no_repeated_game_id_groups"


def test_aux_fetch_build_target_realism_and_aux_retrieval(tmp_path: Path) -> None:
    waterhorse_root = tmp_path / "waterhorse"
    annotated_path = waterhorse_root / "chessgpt_data" / "annotated_pgn" / "annotated_pgn-data.jsonl-00000-of-00001"
    c4_path = waterhorse_root / "chessgpt_data" / "c4" / "c4-data.jsonl-00000-of-00001"
    _write_jsonl(
        annotated_path,
        [
            {
                "metadata": {},
                "text": '[Event "Test"]\n[Site "https://example.com"]\n[Date "2024.01.01"]\n[Round "?"]\n[White "A"]\n[Black "B"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 *',
                "pipeline_key": "annotated_1",
            }
        ],
    )
    _write_jsonl(
        c4_path,
        [
            {
                "metadata": {"url": "https://example.com"},
                "text": "General chess advice without a board anchor.",
                "pipeline_key": "c4_1",
            }
        ],
    )
    text_sample = tmp_path / "chessgpt_text_corpus.jsonl"
    _write_jsonl(
        text_sample,
        [
            {
                "position_id": "sample_1",
                "source": "sample",
                "schema": "text",
                "fen": chess.STARTING_FEN,
                "prompt": "Plan?",
                "response": "Develop pieces.",
            }
        ],
    )

    fetch_manifest = tmp_path / "language_probe_v4" / "manifests" / "aux_fetch_lock.yaml"
    write_aux_fetch_lock(
        entries=[
            record_local_aux_source(
                source_name="waterhorse_chess_data",
                path=waterhorse_root,
                source_url="https://huggingface.co/datasets/Waterhorse/chess_data",
                version="local-test",
                usability="partial",
            ),
            record_local_aux_source(
                source_name="chessgpt_text_sample",
                path=text_sample,
                source_url="local-sample",
                version="local-test",
                usability="usable",
            ),
        ],
        manifest_path=fetch_manifest,
        notes_path=tmp_path / "language_probe_v4" / "manifests" / "aux_fetch_notes.md",
    )
    fetch_payload = yaml.safe_load(fetch_manifest.read_text(encoding="utf-8"))
    assert len(fetch_payload["artifacts"]) == 2
    assert fetch_payload["artifacts"][0]["usability"] == "partial"

    output_root = tmp_path / "language_probe_v4"
    build_result = build_aux_language_corpora(
        source_paths={
            "waterhorse_raw": waterhorse_root,
            "chessgpt_text_sample": text_sample,
        },
        output_root=output_root,
    )
    assert sum(build_result["board_anchored_split_counts"].values()) >= 2
    assert build_result["text_only_count"] >= 1
    aux_manifest = yaml.safe_load((output_root / "manifests" / "aux_source_manifest.yaml").read_text(encoding="utf-8"))
    assert aux_manifest["source_reports"]["waterhorse_raw"]["board_anchored_rows"] >= 1

    probe_root = tmp_path / "language_probe_v3_fix"
    (probe_root / "manifests").mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        _write_jsonl(
            probe_root / f"mate_{split_name}.jsonl",
            [
                {
                    "probe_id": f"mate_{split_name}_1",
                    "source": "mate",
                    "split": split_name,
                    "fen": chess.STARTING_FEN,
                    "strategy_text": "Sacrifice a piece to unlock a file or diagonal in proximity to the opposing king.",
                    "tactic_text": "Trade the lower value piece for a higher value piece.",
                }
            ],
        )
    with (probe_root / "manifests" / "probe_targets_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"label_frequencies_by_source": {"mate": {"check": 10, "king_safety": 8}}},
            handle,
            sort_keys=False,
        )
    mate_keyword_audit_path = tmp_path / "mate_keyword_audit.json"
    mate_keyword_audit_path.write_text(
        json.dumps(
            {
                "candidate_labels": [
                    {
                        "label": "trade_up",
                        "patterns": ["trade the lower value piece for a higher value piece"],
                        "ambiguity_risk": "low",
                        "support_count": 100,
                        "recommended_for_future_v2": True,
                    },
                    {
                        "label": "open_line_attack",
                        "patterns": ["open file or diagonal"],
                        "ambiguity_risk": "low",
                        "support_count": 80,
                        "recommended_for_future_v2": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    realism_result = write_target_realism_report(
        probe_root=probe_root,
        aux_root=output_root,
        mate_keyword_audit_path=mate_keyword_audit_path,
        output_root=output_root,
        create_mate_v2=True,
    )
    assert realism_result["report"]["created_mate_targets_v2"] is True
    assert "trade_up" in realism_result["report"]["selected_v2_labels"]

    corpus_root = tmp_path / "retrieval_corpus"
    embedding_root = tmp_path / "embedding_exports"
    for split_name, row_count in (("train", 4), ("val", 2), ("test", 2)):
        probe_ids = [f"aux_{split_name}_{index}" for index in range(row_count)]
        rows = []
        features = []
        for index, probe_id in enumerate(probe_ids):
            positive = index < max(1, row_count // 2)
            rows.append(
                {
                    "probe_id": probe_id,
                    "text": "Attack the king with forcing moves." if positive else "Improve piece placement quietly.",
                }
            )
            features.append([3.0, 0.0] if positive else [-3.0, 0.0])
        _write_jsonl(corpus_root / f"aux_board_anchored_{split_name}.jsonl", rows)
        for backbone in ("g1", "g3"):
            for seed in (11, 17, 23):
                seed_dir = embedding_root / backbone / f"seed{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                payload = _embedding_payload(probe_ids, split_name, features)
                payload["seed"] = seed
                torch.save(payload, seed_dir / f"aux_board_anchored_{split_name}_embeddings.pt")

    retrieval_result = run_raw_text_retrieval_probes(
        embedding_root=embedding_root,
        corpus_root=corpus_root,
        output_dir=tmp_path / "raw_text_retrieval_v2",
        backbone_seeds=[11, 17, 23],
        mate_min_df=1,
        puzzle_min_df=1,
        max_vocab_size=32,
        families=["aux_board_anchored"],
    )
    assert retrieval_result["aggregate"]
    assert retrieval_result["aggregate"][0]["family"] == "aux_board_anchored"
