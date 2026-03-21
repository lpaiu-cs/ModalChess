"""Raw-text and synthetic-tag retrieval probes on frozen chess embeddings."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_embedding_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"embedding payload가 dict가 아니다: {path}")
    return payload


def _normalize_tag_text(tag: str) -> str:
    tag = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(tag))
    tag = tag.replace("_", " ")
    return tag.lower()


def _mate_text_document(row: dict[str, Any]) -> str:
    return " ".join(
        part.strip()
        for part in (
            str(row.get("strategy_text") or ""),
            str(row.get("tactic_text") or ""),
        )
        if part and part.strip()
    )


def _puzzle_synthetic_document(corpus_row: dict[str, Any], target_row: dict[str, Any]) -> str:
    tags = [_normalize_tag_text(tag) for tag in corpus_row.get("theme_tags") or target_row.get("target_labels") or []]
    flag_tokens = []
    for field_name, token_name in (
        ("promotion_flag", "promotion flag"),
        ("castling_flag", "castling flag"),
        ("en_passant_flag", "en passant flag"),
        ("check_evasion_flag", "check evasion flag"),
    ):
        if target_row.get(field_name):
            flag_tokens.append(token_name)
    return " ".join(tags + flag_tokens)


def _tokenize_text(text: str) -> list[str]:
    tokens = [token for token in re.findall(r"[a-z0-9']+", text.lower()) if len(token) > 1]
    return [token for token in tokens if token not in STOPWORDS]


def _build_vocab(documents: list[str], min_df: int, max_vocab_size: int) -> tuple[list[str], dict[str, int], dict[str, float]]:
    doc_freq: dict[str, int] = {}
    for document in documents:
        seen = set(_tokenize_text(document))
        for token in seen:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    filtered = [
        token
        for token, count in sorted(doc_freq.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_df
    ][:max_vocab_size]
    idf = {
        token: 1.0 + math.log((1.0 + len(documents)) / (1.0 + doc_freq[token]))
        for token in filtered
    }
    token_to_index = {token: index for index, token in enumerate(filtered)}
    return filtered, token_to_index, idf


def _tfidf_matrix(documents: list[str], token_to_index: dict[str, int], idf: dict[str, float]) -> torch.Tensor:
    matrix = torch.zeros((len(documents), len(token_to_index)), dtype=torch.float32)
    index_to_token = {index: token for token, index in token_to_index.items()}
    for row_index, document in enumerate(documents):
        tokens = _tokenize_text(document)
        if not tokens:
            continue
        counts: dict[int, int] = {}
        for token in tokens:
            token_index = token_to_index.get(token)
            if token_index is None:
                continue
            counts[token_index] = counts.get(token_index, 0) + 1
        token_count = sum(counts.values())
        if token_count == 0:
            continue
        for token_index, count in counts.items():
            token = index_to_token[token_index]
            matrix[row_index, token_index] = (count / token_count) * float(idf[token])
    return matrix


def _normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    norms = matrix.norm(dim=1, keepdim=True)
    norms = torch.where(norms < 1e-8, torch.ones_like(norms), norms)
    return matrix / norms


def _standardize_features(
    train_features: torch.Tensor,
    val_features: torch.Tensor,
    test_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (
        (train_features - mean) / std,
        (val_features - mean) / std,
        (test_features - mean) / std,
    )


class LinearTextProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(inputs)


class MlpTextProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        hidden_dim = max(64, min(256, input_dim * 2))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _subsample_tensors(features: torch.Tensor, targets: torch.Tensor, limit: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if features.size(0) <= limit:
        return features, targets
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(features.size(0), generator=generator)[:limit]
    return features.index_select(0, indices), targets.index_select(0, indices)


def _train_text_probe(
    *,
    model_kind: str,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    seed: int,
    max_train_rows: int | None = None,
) -> tuple[nn.Module, float]:
    if max_train_rows is not None:
        train_features, train_targets = _subsample_tensors(train_features, train_targets, max_train_rows, seed)
    input_dim = train_features.size(1)
    output_dim = train_targets.size(1)
    if model_kind == "linear":
        model: nn.Module = LinearTextProbe(input_dim=input_dim, output_dim=output_dim)
        learning_rate = 5e-3
        max_epochs = 12
    elif model_kind == "mlp":
        model = MlpTextProbe(input_dim=input_dim, output_dim=output_dim)
        learning_rate = 1e-3
        max_epochs = 16
    else:
        raise ValueError(f"지원하지 않는 probe model: {model_kind}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(TensorDataset(train_features, train_targets), batch_size=1024, shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_alignment = float("-inf")
    patience = 3
    patience_left = patience
    for _epoch in range(max_epochs):
        model.train()
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_predictions = _normalize_rows(model(val_features))
            val_alignment = float((val_predictions * _normalize_rows(val_targets)).sum(dim=1).mean().item())
        if val_alignment > best_val_alignment:
            best_val_alignment = val_alignment
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), best_val_alignment


def _chunked_retrieval_metrics(query_vectors: torch.Tensor, key_vectors: torch.Tensor, chunk_size: int = 256) -> tuple[float, float, float]:
    if query_vectors.numel() == 0 or key_vectors.numel() == 0:
        return 0.0, 0.0, 0.0
    total = query_vectors.size(0)
    recall_at_1 = 0.0
    recall_at_5 = 0.0
    reciprocal_rank_sum = 0.0
    for start in range(0, total, chunk_size):
        query_chunk = query_vectors[start : start + chunk_size]
        scores = query_chunk @ key_vectors.transpose(0, 1)
        diagonal_indices = torch.arange(start, min(start + chunk_size, total), dtype=torch.long)
        local_indices = torch.arange(diagonal_indices.numel(), dtype=torch.long)
        target_scores = scores[local_indices, diagonal_indices]
        ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1
        recall_at_1 += float((ranks == 1).sum().item())
        recall_at_5 += float((ranks <= 5).sum().item())
        reciprocal_rank_sum += float((1.0 / ranks.float()).sum().item())
    count = float(total)
    return recall_at_1 / count, recall_at_5 / count, reciprocal_rank_sum / count


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
        "val_alignment",
    )
    for key in sorted(grouped):
        family, backbone, pool, probe_model = key
        group_rows = sorted(grouped[key], key=lambda row: int(row["seed"]))
        aggregate_row: dict[str, Any] = {
            "family": family,
            "backbone": backbone,
            "pool": pool,
            "probe_model": probe_model,
            "text_side_kind": group_rows[0]["text_side_kind"],
            "seed_count": len(group_rows),
            "seeds": [int(row["seed"]) for row in group_rows],
            "vocab_size": int(group_rows[0]["vocab_size"]),
            "test_rows_mean": statistics.fmean(float(row["test_rows"]) for row in group_rows),
        }
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in group_rows]
            aggregate_row[f"{metric_name}_mean"] = statistics.fmean(values)
            aggregate_row[f"{metric_name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _summary_markdown(aggregate_rows: list[dict[str, Any]]) -> str:
    lines = ["# Raw-Text Retrieval Summary", ""]
    lines.append("- MATE uses real strategy/tactic text retrieval.")
    lines.append("- Puzzle uses synthetic tag-string retrieval from themes and special-rule flags.")
    lines.append("- These are evaluation-only retrieval probes, not language-fusion results.")
    lines.append("")
    for family in ("mate", "puzzle"):
        family_rows = [row for row in aggregate_rows if row["family"] == family]
        if not family_rows:
            continue
        best_row = max(family_rows, key=lambda row: row["board_to_text_mrr_mean"])
        random_r1 = 1.0 / max(int(best_row["test_rows_mean"]), 1)
        random_r5 = min(5.0 / max(int(best_row["test_rows_mean"]), 1), 1.0)
        lines.append(
            f"- `{family}` best: text_side={best_row['text_side_kind']}, backbone={best_row['backbone']}, pool={best_row['pool']}, "
            f"probe={best_row['probe_model']}, R@1={best_row['board_to_text_recall_at_1_mean']:.4f} +/- {best_row['board_to_text_recall_at_1_std']:.4f}, "
            f"R@5={best_row['board_to_text_recall_at_5_mean']:.4f} +/- {best_row['board_to_text_recall_at_5_std']:.4f}, "
            f"MRR={best_row['board_to_text_mrr_mean']:.4f} +/- {best_row['board_to_text_mrr_std']:.4f}"
        )
        lines.append(
            f"  random_baseline_approx: R@1={random_r1:.6f}, R@5={random_r5:.6f}"
        )
    return "\n".join(lines) + "\n"


def _align_rows_by_probe_id(
    corpus_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    if target_rows is None:
        return corpus_rows, None
    by_probe_id = {str(row["probe_id"]): row for row in corpus_rows}
    aligned_corpus_rows: list[dict[str, Any]] = []
    aligned_target_rows: list[dict[str, Any]] = []
    for target_row in target_rows:
        probe_id = str(target_row["probe_id"])
        corpus_row = by_probe_id.get(probe_id)
        if corpus_row is None:
            continue
        aligned_corpus_rows.append(corpus_row)
        aligned_target_rows.append(target_row)
    return aligned_corpus_rows, aligned_target_rows


def _documents_for_family(
    family: str,
    corpus_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]] | None,
) -> list[str]:
    if family == "mate":
        return [_mate_text_document(row) for row in corpus_rows]
    if target_rows is None:
        raise ValueError("puzzle retrieval에는 target rows가 필요하다.")
    return [
        _puzzle_synthetic_document(corpus_row, target_row)
        for corpus_row, target_row in zip(corpus_rows, target_rows, strict=True)
    ]


def run_raw_text_retrieval_probes(
    *,
    embedding_root: str | Path = "outputs/week6/embedding_exports",
    corpus_root: str | Path = "data/pilot/language_probe_v2",
    output_dir: str | Path = "outputs/week7/raw_text_retrieval",
    backbone_seeds: list[int] | None = None,
    mate_min_df: int = 50,
    puzzle_min_df: int = 25,
    max_vocab_size: int = 256,
) -> dict[str, Any]:
    """Run raw-text/synthetic-tag retrieval probes on frozen embeddings."""
    seed_list = backbone_seeds or [11, 17, 23]
    embedding_root_path = Path(embedding_root)
    corpus_root_path = Path(corpus_root)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for family, min_df, text_side_kind in (
        ("mate", mate_min_df, "raw_text"),
        ("puzzle", puzzle_min_df, "synthetic_tag_string"),
    ):
        corpus_rows_by_split = {
            split_name: _load_jsonl(corpus_root_path / f"{family}_{split_name}.jsonl")
            for split_name in ("train", "val", "test")
        }
        target_rows_by_split = {
            split_name: _load_jsonl(corpus_root_path / f"{family}_targets_{split_name}.jsonl")
            if (corpus_root_path / f"{family}_targets_{split_name}.jsonl").exists()
            else None
            for split_name in ("train", "val", "test")
        }
        aligned_rows_by_split: dict[str, list[dict[str, Any]]] = {}
        aligned_targets_by_split: dict[str, list[dict[str, Any]] | None] = {}
        for split_name in ("train", "val", "test"):
            aligned_rows, aligned_targets = _align_rows_by_probe_id(
                corpus_rows_by_split[split_name],
                target_rows_by_split[split_name],
            )
            aligned_rows_by_split[split_name] = aligned_rows
            aligned_targets_by_split[split_name] = aligned_targets
        documents_by_split = {
            split_name: _documents_for_family(family, aligned_rows_by_split[split_name], aligned_targets_by_split[split_name])
            for split_name in ("train", "val", "test")
        }
        vocab, token_to_index, idf = _build_vocab(
            documents_by_split["train"],
            min_df=min_df,
            max_vocab_size=max_vocab_size,
        )
        if not vocab:
            continue
        tfidf_by_split = {
            split_name: _normalize_rows(_tfidf_matrix(documents_by_split[split_name], token_to_index, idf))
            for split_name in ("train", "val", "test")
        }

        probe_ids_by_split = {
            split_name: [str(row["probe_id"]) for row in aligned_rows_by_split[split_name]]
            for split_name in ("train", "val", "test")
        }
        for backbone_name in ("g1", "g3"):
            for seed in seed_list:
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
                        model, val_alignment = _train_text_probe(
                            model_kind=probe_model,
                            train_features=train_features,
                            train_targets=tfidf_by_split["train"],
                            val_features=val_features,
                            val_targets=tfidf_by_split["val"],
                            seed=seed,
                            max_train_rows=train_limit,
                        )
                        model.eval()
                        with torch.no_grad():
                            predicted_test = _normalize_rows(model(test_features))
                        board_to_text = _chunked_retrieval_metrics(predicted_test, tfidf_by_split["test"])
                        text_to_board = _chunked_retrieval_metrics(tfidf_by_split["test"], predicted_test)
                        results.append(
                            {
                                "family": family,
                                "text_side_kind": text_side_kind,
                                "backbone": backbone_name,
                                "seed": seed,
                                "pool": pool_name,
                                "probe_model": probe_model,
                                "vocab_size": len(vocab),
                                "test_rows": int(test_features.size(0)),
                                "val_alignment": val_alignment,
                                "board_to_text_recall_at_1": board_to_text[0],
                                "board_to_text_recall_at_5": board_to_text[1],
                                "board_to_text_mrr": board_to_text[2],
                                "text_to_board_recall_at_1": text_to_board[0],
                                "text_to_board_recall_at_5": text_to_board[1],
                                "text_to_board_mrr": text_to_board[2],
                            }
                        )

    aggregate_rows = _aggregate_results(results)
    json_path = output_root / "raw_text_retrieval_results.json"
    csv_path = output_root / "raw_text_retrieval_results.csv"
    summary_path = output_root / "raw_text_retrieval_summary.md"
    json_path.write_text(json.dumps({"results": results, "aggregate": aggregate_rows}, indent=2), encoding="utf-8")
    _write_csv(csv_path, results)
    summary_path.write_text(_summary_markdown(aggregate_rows), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "results": results,
        "aggregate": aggregate_rows,
    }
