"""Retrieval-style frozen-backbone language readiness probes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
from typing import Any

import torch

from modalchess.data.probe_targets import MATE_KEYWORD_MAP
from modalchess.eval.language_readiness import (
    _filter_label_vocab,
    _load_embedding_payload,
    _load_target_rows,
    _standardize_features,
    _train_probe,
)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _normalize_token(token: str) -> str:
    return token.strip().lower().replace(" ", "_")


def _puzzle_text_tokens(target_row: dict[str, Any]) -> list[str]:
    tokens = [_normalize_token(str(label)) for label in target_row.get("target_labels", [])]
    for field_name, token_name in (
        ("promotion_flag", "promotion_flag"),
        ("castling_flag", "castling_flag"),
        ("en_passant_flag", "en_passant_flag"),
        ("check_evasion_flag", "check_evasion_flag"),
    ):
        if target_row.get(field_name):
            tokens.append(token_name)
    return tokens


def _mate_text_tokens(probe_row: dict[str, Any]) -> list[str]:
    text = " ".join(str(probe_row.get(key) or "") for key in ("strategy_text", "tactic_text")).lower()
    tokens: list[str] = []
    for label_name, keywords in MATE_KEYWORD_MAP.items():
        if any(keyword in text for keyword in keywords):
            tokens.append(_normalize_token(label_name))
    return tokens


def _build_retrieval_vocab(
    family: str,
    train_probe_rows: list[dict[str, Any]],
    train_target_rows: list[dict[str, Any]],
    min_train_positive: int,
) -> list[str]:
    if family == "mate":
        train_rows_for_vocab = [
            {
                "target_labels": _mate_text_tokens(probe_row),
            }
            for probe_row in train_probe_rows
        ]
        return _filter_label_vocab(train_rows_for_vocab, min_train_positive=min_train_positive)

    train_rows_for_vocab = []
    for target_row in train_target_rows:
        train_rows_for_vocab.append(
            {
                "target_labels": _puzzle_text_tokens(target_row),
            }
        )
    return _filter_label_vocab(train_rows_for_vocab, min_train_positive=min_train_positive)


def _matrix_from_tokens(rows: list[list[str]], vocab: list[str]) -> torch.Tensor:
    token_to_index = {token: index for index, token in enumerate(vocab)}
    matrix = torch.zeros((len(rows), len(vocab)), dtype=torch.float32)
    for row_index, row_tokens in enumerate(rows):
        for token in row_tokens:
            if token in token_to_index:
                matrix[row_index, token_to_index[token]] = 1.0
    return matrix


def _aligned_split_rows(
    *,
    family: str,
    corpus_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corpus_by_probe_id = {str(row["probe_id"]): row for row in corpus_rows}
    aligned_probe_rows: list[dict[str, Any]] = []
    aligned_target_rows: list[dict[str, Any]] = []
    for target_row in target_rows:
        probe_id = str(target_row["probe_id"])
        probe_row = corpus_by_probe_id.get(probe_id)
        if probe_row is None:
            continue
        aligned_probe_rows.append(probe_row)
        aligned_target_rows.append(target_row)
    return aligned_probe_rows, aligned_target_rows


def _retrieval_target_matrix(
    family: str,
    probe_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    vocab: list[str],
) -> torch.Tensor:
    if family == "mate":
        tokens = [_mate_text_tokens(probe_row) for probe_row in probe_rows]
    else:
        tokens = [_puzzle_text_tokens(target_row) for target_row in target_rows]
    return _matrix_from_tokens(tokens, vocab)


def _normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    norms = matrix.norm(dim=1, keepdim=True)
    norms = torch.where(norms < 1e-8, torch.ones_like(norms), norms)
    return matrix / norms


def _retrieval_metrics(similarity: torch.Tensor) -> dict[str, float]:
    if similarity.numel() == 0:
        return {
            "board_to_text_recall_at_1": 0.0,
            "board_to_text_recall_at_5": 0.0,
            "board_to_text_mrr": 0.0,
            "text_to_board_recall_at_1": 0.0,
            "text_to_board_recall_at_5": 0.0,
            "text_to_board_mrr": 0.0,
        }

    def _directional_metrics(scores: torch.Tensor) -> tuple[float, float, float]:
        topk = min(5, scores.size(1))
        recall_at_1 = 0.0
        recall_at_5 = 0.0
        reciprocal_ranks: list[float] = []
        for row_index in range(scores.size(0)):
            ranking = torch.argsort(scores[row_index], descending=True)
            target_rank = int((ranking == row_index).nonzero(as_tuple=False)[0].item()) + 1
            recall_at_1 += float(target_rank == 1)
            recall_at_5 += float(target_rank <= topk)
            reciprocal_ranks.append(1.0 / target_rank)
        count = float(scores.size(0))
        return recall_at_1 / count, recall_at_5 / count, statistics.fmean(reciprocal_ranks)

    board_to_text = _directional_metrics(similarity)
    text_to_board = _directional_metrics(similarity.transpose(0, 1))
    return {
        "board_to_text_recall_at_1": board_to_text[0],
        "board_to_text_recall_at_5": board_to_text[1],
        "board_to_text_mrr": board_to_text[2],
        "text_to_board_recall_at_1": text_to_board[0],
        "text_to_board_recall_at_5": text_to_board[1],
        "text_to_board_mrr": text_to_board[2],
    }


def _aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["backbone"]), str(row["pool"]), str(row["probe_model"]))
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    metric_names = (
        "board_to_text_recall_at_1",
        "board_to_text_recall_at_5",
        "board_to_text_mrr",
        "text_to_board_recall_at_1",
        "text_to_board_recall_at_5",
        "text_to_board_mrr",
    )
    for key in sorted(grouped):
        family, backbone, pool, probe_model = key
        group_rows = sorted(grouped[key], key=lambda row: int(row["seed"]))
        aggregate_row: dict[str, Any] = {
            "family": family,
            "backbone": backbone,
            "pool": pool,
            "probe_model": probe_model,
            "seed_count": len(group_rows),
            "seeds": [int(row["seed"]) for row in group_rows],
            "label_count": int(group_rows[0]["label_count"]),
            "test_rows_mean": statistics.fmean(float(row["test_rows"]) for row in group_rows),
        }
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in group_rows]
            aggregate_row[f"{metric_name}_mean"] = statistics.fmean(values)
            aggregate_row[f"{metric_name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(aggregate_rows: list[dict[str, Any]]) -> str:
    lines = ["# Retrieval Readiness Summary", ""]
    lines.append(
        "- This is a retrieval-style readiness probe over lexical bag-of-words targets, not a language-fusion result."
    )
    lines.append("")
    for family in ("puzzle", "mate"):
        family_rows = [row for row in aggregate_rows if row["family"] == family]
        if not family_rows:
            continue
        best_row = max(family_rows, key=lambda row: row["board_to_text_mrr_mean"])
        random_r1 = 1.0 / max(int(best_row["test_rows_mean"]), 1)
        random_r5 = min(5.0 / max(int(best_row["test_rows_mean"]), 1), 1.0)
        lines.append(
            f"- `{family}` best: backbone={best_row['backbone']}, pool={best_row['pool']}, "
            f"probe={best_row['probe_model']}, R@1={best_row['board_to_text_recall_at_1_mean']:.4f} +/- {best_row['board_to_text_recall_at_1_std']:.4f}, "
            f"R@5={best_row['board_to_text_recall_at_5_mean']:.4f} +/- {best_row['board_to_text_recall_at_5_std']:.4f}, "
            f"MRR={best_row['board_to_text_mrr_mean']:.4f} +/- {best_row['board_to_text_mrr_std']:.4f}"
        )
        lines.append(
            f"  random_baseline_approx: R@1={random_r1:.6f}, R@5={random_r5:.6f}"
        )
    return "\n".join(lines) + "\n"


def run_retrieval_readiness_probes(
    *,
    embedding_root: str | Path,
    corpus_root: str | Path,
    target_root: str | Path,
    output_dir: str | Path,
    backbone_seeds: list[int],
    mate_min_train_positive: int = 25,
    puzzle_min_train_positive: int = 25,
) -> dict[str, Any]:
    """Run retrieval-style readiness probes on frozen board embeddings."""
    embedding_root_path = Path(embedding_root)
    corpus_root_path = Path(corpus_root)
    target_root_path = Path(target_root)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for family, min_train_positive in (("mate", mate_min_train_positive), ("puzzle", puzzle_min_train_positive)):
        corpus_rows_by_split = {
            split_name: _load_jsonl(corpus_root_path / f"{family}_{split_name}.jsonl")
            for split_name in ("train", "val", "test")
        }
        target_rows_by_split = {
            split_name: _load_target_rows(target_root_path / f"{family}_targets_{split_name}.jsonl")
            for split_name in ("train", "val", "test")
        }
        aligned_probe_rows_by_split: dict[str, list[dict[str, Any]]] = {}
        aligned_target_rows_by_split: dict[str, list[dict[str, Any]]] = {}
        for split_name in ("train", "val", "test"):
            aligned_probe_rows, aligned_target_rows = _aligned_split_rows(
                family=family,
                corpus_rows=corpus_rows_by_split[split_name],
                target_rows=target_rows_by_split[split_name],
            )
            aligned_probe_rows_by_split[split_name] = aligned_probe_rows
            aligned_target_rows_by_split[split_name] = aligned_target_rows

        label_vocab = _build_retrieval_vocab(
            family,
            aligned_probe_rows_by_split["train"],
            aligned_target_rows_by_split["train"],
            min_train_positive=min_train_positive,
        )
        if not label_vocab:
            continue

        target_matrices = {
            split_name: _retrieval_target_matrix(
                family,
                aligned_probe_rows_by_split[split_name],
                aligned_target_rows_by_split[split_name],
                label_vocab,
            )
            for split_name in ("train", "val", "test")
        }
        probe_ids_by_split = {
            split_name: [str(row["probe_id"]) for row in aligned_target_rows_by_split[split_name]]
            for split_name in ("train", "val", "test")
        }
        text_matrix_test = _normalize_rows(target_matrices["test"])

        for backbone_name in ("g1", "g3"):
            for seed in backbone_seeds:
                embedding_dir = embedding_root_path / backbone_name / f"seed{seed}"
                embedding_payloads = {
                    split_name: _load_embedding_payload(embedding_dir / f"{family}_{split_name}_embeddings.pt")
                    for split_name in ("train", "val", "test")
                }
                for pool_name in ("board_pooled", "context_pooled"):
                    split_features: dict[str, torch.Tensor] = {}
                    for split_name in ("train", "val", "test"):
                        payload = embedding_payloads[split_name]
                        probe_id_to_index = {str(probe_id): index for index, probe_id in enumerate(payload["probe_id"])}
                        ordered_indices = torch.tensor(
                            [probe_id_to_index[probe_id] for probe_id in probe_ids_by_split[split_name]],
                            dtype=torch.long,
                        )
                        split_features[split_name] = payload[pool_name].index_select(0, ordered_indices).float()

                    train_features, val_features, test_features = _standardize_features(
                        split_features["train"],
                        split_features["val"],
                        split_features["test"],
                    )

                    for probe_model, train_limit in (("linear", None), ("mlp", 100000 if family == "mate" else 50000)):
                        trained_model, _ = _train_probe(
                            model_kind=probe_model,
                            train_features=train_features,
                            train_targets=target_matrices["train"],
                            val_features=val_features,
                            val_targets=target_matrices["val"],
                            seed=seed,
                            max_train_rows=train_limit,
                        )
                        trained_model.eval()
                        with torch.no_grad():
                            test_logits = trained_model(test_features)
                        query_vectors = _normalize_rows(torch.sigmoid(test_logits))
                        similarity = query_vectors @ text_matrix_test.transpose(0, 1)
                        results.append(
                            {
                                "family": family,
                                "backbone": backbone_name,
                                "seed": seed,
                                "pool": pool_name,
                                "probe_model": probe_model,
                                "label_count": len(label_vocab),
                                "test_rows": int(test_features.size(0)),
                                **_retrieval_metrics(similarity),
                            }
                        )

    aggregate_rows = _aggregate_results(results)
    summary_path = output_root / "retrieval_summary.md"
    json_path = output_root / "retrieval_results.json"
    csv_path = output_root / "retrieval_results.csv"
    summary_path.write_text(_summary_markdown(aggregate_rows), encoding="utf-8")
    json_path.write_text(
        json.dumps({"results": results, "aggregate": aggregate_rows}, indent=2),
        encoding="utf-8",
    )
    _write_csv(csv_path, results)
    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "results": results,
        "aggregate": aggregate_rows,
    }
