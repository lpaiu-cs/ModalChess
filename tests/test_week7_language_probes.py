from __future__ import annotations

import json
from pathlib import Path

import chess
import torch
import yaml

from modalchess.data.aux_language import audit_aux_language_sources, build_aux_language_corpora
from modalchess.data.mate_keyword_audit import audit_mate_keyword_coverage
from modalchess.data.text_realism_audit import audit_probe_text_realism
from modalchess.eval.raw_text_retrieval import run_raw_text_retrieval_probes


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


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


def test_aux_language_audit_and_build(tmp_path: Path) -> None:
    text_path = tmp_path / "chessgpt_text_corpus.jsonl"
    conversation_path = tmp_path / "chessgpt_conversation_corpus.jsonl"
    _write_jsonl(
        text_path,
        [
            {
                "position_id": "text_row_1",
                "source": "waterhorse_chess_data",
                "schema": "text",
                "fen": chess.STARTING_FEN,
                "candidate_moves": [],
                "prompt": "What is the plan?",
                "response": "Fight for the center.",
                "text": None,
            },
            {
                "position_id": "text_row_2",
                "source": "waterhorse_chess_data",
                "schema": "text",
                "fen": None,
                "candidate_moves": [],
                "prompt": "General chess advice?",
                "response": "Develop pieces quickly.",
                "text": None,
            },
        ],
    )
    _write_jsonl(
        conversation_path,
        [
            {
                "position_id": "conv_row_1",
                "source": "waterhorse_chess_data",
                "schema": "conversation",
                "fen": "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
                "candidate_moves": [],
                "messages": [
                    {"role": "user", "content": "How should White respond?"},
                    {"role": "assistant", "content": "White should move the king."},
                ],
            }
        ],
    )

    source_paths = {
        "waterhorse_raw": tmp_path / "missing_waterhorse",
        "chessgpt_text_sample": text_path,
        "chessgpt_conversation_sample": conversation_path,
    }
    audit_result = audit_aux_language_sources(
        source_paths=source_paths,
        output_dir=tmp_path / "reports",
    )
    build_result = build_aux_language_corpora(
        source_paths=source_paths,
        output_root=tmp_path / "language_probe_v3",
    )
    assert audit_result["report"]["sources"]["chessgpt_text_sample"]["board_anchored_rows"] == 1
    assert audit_result["report"]["sources"]["chessgpt_text_sample"]["text_only_rows"] == 1
    assert build_result["board_anchored_split_counts"]["train"] + build_result["board_anchored_split_counts"]["val"] + build_result["board_anchored_split_counts"]["test"] == 2
    assert build_result["text_only_count"] == 1
    manifest = yaml.safe_load((tmp_path / "language_probe_v3" / "manifests" / "aux_source_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["text_only_count"] == 1


def test_text_realism_and_mate_keyword_audit(tmp_path: Path) -> None:
    probe_root = tmp_path / "language_probe_v2"
    (probe_root / "manifests").mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        probe_root / "mate_train.jsonl",
        [
            {
                "probe_id": "mate_1",
                "strategy_text": "Sacrifice a piece to unlock a file or diagonal in proximity to the opposing king.",
                "tactic_text": "Trade the lower value piece for a higher value piece.",
            }
        ],
    )
    _write_jsonl(
        probe_root / "puzzle_train.jsonl",
        [
            {
                "probe_id": "puzzle_1",
                "theme_tags": ["fork"],
            }
        ],
    )
    for split_name in ("val", "test"):
        _write_jsonl(probe_root / f"mate_{split_name}.jsonl", [])
        _write_jsonl(probe_root / f"puzzle_{split_name}.jsonl", [])
    _write_jsonl(
        probe_root / "mate_targets_train.jsonl",
        [{"probe_id": "mate_1", "target_labels": ["check", "king_safety"]}],
    )
    _write_jsonl(
        probe_root / "puzzle_targets_train.jsonl",
        [{"probe_id": "puzzle_1", "target_labels": ["fork"], "promotion_flag": False, "castling_flag": False, "en_passant_flag": False, "check_evasion_flag": False}],
    )
    for split_name in ("val", "test"):
        _write_jsonl(probe_root / f"mate_targets_{split_name}.jsonl", [])
        _write_jsonl(probe_root / f"puzzle_targets_{split_name}.jsonl", [])
    with (probe_root / "manifests" / "probe_targets_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "label_frequencies_by_source": {
                    "mate": {"check": 10, "king_safety": 8},
                    "puzzle": {"fork": 12},
                }
            },
            handle,
            sort_keys=False,
        )
    readiness_path = tmp_path / "probe_results.json"
    retrieval_path = tmp_path / "retrieval_results.json"
    readiness_path.write_text(
        json.dumps(
            {
                "aggregate": [
                    {"family": "mate", "probe_model": "mlp", "test_micro_average_precision_mean": 0.2, "backbone": "g3"},
                    {"family": "puzzle", "probe_model": "mlp", "test_micro_average_precision_mean": 0.4, "backbone": "g1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    retrieval_path.write_text(
        json.dumps(
            {
                "aggregate": [
                    {"family": "mate", "board_to_text_mrr_mean": 0.01, "backbone": "g1"},
                    {"family": "puzzle", "board_to_text_mrr_mean": 0.02, "backbone": "g3"},
                ]
            }
        ),
        encoding="utf-8",
    )

    realism_result = audit_probe_text_realism(
        input_root=probe_root,
        week6_readiness_path=readiness_path,
        week6_retrieval_path=retrieval_path,
        output_dir=tmp_path / "reports",
        retrieval_min_support=1,
    )
    assert realism_result["report"]["sources"]["mate"]["natural_text_available"] is True
    assert realism_result["report"]["sources"]["puzzle"]["uses_raw_text_in_week6_retrieval"] is False

    mate_input = tmp_path / "language_mate.jsonl"
    _write_jsonl(
        mate_input,
        [
            {
                "strategy_text": "Sacrifice a piece to unlock a file or diagonal in proximity to the opposing king.",
                "tactic_text": "Trade the lower value piece for a higher value piece.",
            }
        ],
    )
    keyword_result = audit_mate_keyword_coverage(
        input_path=mate_input,
        output_dir=tmp_path / "reports",
        min_support=1,
    )
    candidate_support = {row["label"]: row["support_count"] for row in keyword_result["report"]["candidate_labels"]}
    assert candidate_support["trade_up"] > 0
    assert candidate_support["open_line_attack"] > 0


def test_raw_text_retrieval_probe_smoke(tmp_path: Path) -> None:
    corpus_root = tmp_path / "language_probe_v2"
    embedding_root = tmp_path / "embedding_exports"
    output_dir = tmp_path / "raw_text_retrieval"
    split_rows = {
        "train": 4,
        "val": 2,
        "test": 2,
    }
    for family in ("mate", "puzzle"):
        for split_name, row_count in split_rows.items():
            corpus_rows: list[dict[str, object]] = []
            target_rows: list[dict[str, object]] = []
            probe_ids = [f"{family}_{split_name}_{index}" for index in range(row_count)]
            features = []
            for index, probe_id in enumerate(probe_ids):
                positive = index < max(1, row_count // 2)
                features.append([3.0, 0.0] if positive else [-3.0, 0.0])
                if family == "mate":
                    corpus_rows.append(
                        {
                            "probe_id": probe_id,
                            "strategy_text": "Attack the king with forcing moves." if positive else "Improve piece placement quietly.",
                            "tactic_text": "Check follows." if positive else "Quiet maneuver.",
                        }
                    )
                else:
                    corpus_rows.append(
                        {
                            "probe_id": probe_id,
                            "theme_tags": ["fork", "mate"] if positive else ["endgame", "quietMove"],
                        }
                    )
                    target_rows.append(
                        {
                            "probe_id": probe_id,
                            "target_labels": ["fork", "mate"] if positive else ["endgame", "quietMove"],
                            "promotion_flag": False,
                            "castling_flag": False,
                            "en_passant_flag": False,
                            "check_evasion_flag": False,
                        }
                    )
            _write_jsonl(corpus_root / f"{family}_{split_name}.jsonl", corpus_rows)
            if family == "puzzle":
                _write_jsonl(corpus_root / f"{family}_targets_{split_name}.jsonl", target_rows)
            for backbone in ("g1", "g3"):
                for seed in (11, 17, 23):
                    seed_dir = embedding_root / backbone / f"seed{seed}"
                    seed_dir.mkdir(parents=True, exist_ok=True)
                    payload = _embedding_payload(probe_ids, split_name, features)
                    payload["seed"] = seed
                    torch.save(payload, seed_dir / f"{family}_{split_name}_embeddings.pt")

    result = run_raw_text_retrieval_probes(
        embedding_root=embedding_root,
        corpus_root=corpus_root,
        output_dir=output_dir,
        backbone_seeds=[11, 17, 23],
        mate_min_df=1,
        puzzle_min_df=1,
        max_vocab_size=32,
    )
    assert result["aggregate"]
    assert (output_dir / "raw_text_retrieval_summary.md").exists()
    assert max(row["board_to_text_mrr_mean"] for row in result["aggregate"]) > 0.5
