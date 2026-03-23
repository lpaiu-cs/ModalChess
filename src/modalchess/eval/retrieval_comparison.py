"""Paired comparison utilities for week-17 retrieval variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch

from modalchess.eval.raw_text_retrieval import (
    _align_rows_by_probe_id,
    _build_vocab,
    _documents_for_family,
    _load_embedding_payload,
    _load_jsonl,
    _normalize_rows,
    _standardize_features,
    _tfidf_matrix,
    _train_text_probe,
)


@dataclass(slots=True)
class RetrievalComparisonConfig:
    """Configuration for paired retrieval comparison."""

    mate_min_df: int = 25
    max_vocab_size: int = 512
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 1234


def _strict_reciprocal_ranks(
    query_vectors: torch.Tensor,
    key_vectors: torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> torch.Tensor:
    if query_vectors.numel() == 0 or key_vectors.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    scores = query_vectors @ key_vectors.transpose(0, 1)
    target_scores = scores.diag().unsqueeze(1)
    tie_mask = torch.isclose(scores, target_scores, atol=atol, rtol=rtol)
    ranks = (scores > target_scores).sum(dim=1) + tie_mask.sum(dim=1)
    return 1.0 / ranks.float()


def _bootstrap_delta_ci(deltas: torch.Tensor, *, samples: int, seed: int) -> tuple[float, float]:
    if deltas.numel() == 0:
        return 0.0, 0.0
    generator = torch.Generator().manual_seed(seed)
    count = deltas.numel()
    bootstrap_means = torch.empty(samples, dtype=torch.float32)
    for index in range(samples):
        sample_indices = torch.randint(0, count, (count,), generator=generator)
        bootstrap_means[index] = deltas.index_select(0, sample_indices).mean()
    sorted_means, _ = torch.sort(bootstrap_means)
    low_index = max(0, int(0.025 * (samples - 1)))
    high_index = min(samples - 1, int(0.975 * (samples - 1)))
    return float(sorted_means[low_index].item()), float(sorted_means[high_index].item())


def _approx_two_sided_sign_pvalue(positive: int, negative: int) -> float:
    count = positive + negative
    if count == 0:
        return 1.0
    wins = min(positive, negative)
    mean = count * 0.5
    variance = count * 0.25
    z_score = (wins + 0.5 - mean) / math.sqrt(variance)
    tail = 0.5 * math.erfc(abs(z_score) / math.sqrt(2.0))
    return min(1.0, 2.0 * tail)


def _ci_interpretation(ci_low: float, ci_high: float) -> str:
    if ci_low > 0.0:
        return "credible_positive"
    if ci_high < 0.0:
        return "credible_negative"
    return "inconclusive"


def _best_row(aggregate_rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    return max(
        aggregate_rows,
        key=lambda row: (float(row[metric_name]), -float(row["test_rows_mean"]), str(row["backbone"])),
    )


def _common_test_probe_ids(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
) -> list[str]:
    left_ids = {str(row["probe_id"]) for row in left_rows}
    right_ids = {str(row["probe_id"]) for row in right_rows}
    return sorted(left_ids & right_ids)


def _variant_query_scores(
    *,
    variant_name: str,
    results_root: Path,
    embedding_root: Path,
    backbone: str,
    pool: str,
    probe_model: str,
    common_test_ids: list[str],
    seed_list: list[int],
    config: RetrievalComparisonConfig,
) -> dict[str, torch.Tensor | int]:
    corpus_root = results_root / variant_name / "probe_subset"
    split_rows = {
        split_name: _load_jsonl(corpus_root / f"annotated_sidecar_{split_name}.jsonl")
        for split_name in ("train", "val", "test")
    }
    for split_name in ("train", "val", "test"):
        aligned_rows, _ = _align_rows_by_probe_id(split_rows[split_name], None)
        split_rows[split_name] = aligned_rows

    documents_by_split = {
        split_name: _documents_for_family("annotated_sidecar", split_rows[split_name], None)
        for split_name in ("train", "val", "test")
    }
    vocab, token_to_index, idf = _build_vocab(
        documents_by_split["train"],
        min_df=config.mate_min_df,
        max_vocab_size=config.max_vocab_size,
    )
    if not vocab:
        raise ValueError(f"variant {variant_name} produced an empty vocabulary")
    tfidf_by_split = {
        split_name: _normalize_rows(_tfidf_matrix(documents_by_split[split_name], token_to_index, idf))
        for split_name in ("train", "val", "test")
    }
    probe_ids_by_split = {
        split_name: [str(row["probe_id"]) for row in split_rows[split_name]]
        for split_name in ("train", "val", "test")
    }
    common_test_index = {
        probe_id: index
        for index, probe_id in enumerate(probe_ids_by_split["test"])
        if probe_id in set(common_test_ids)
    }
    ordered_common_indices = torch.tensor(
        [common_test_index[probe_id] for probe_id in common_test_ids],
        dtype=torch.long,
    )

    board_rr_by_seed: list[torch.Tensor] = []
    text_rr_by_seed: list[torch.Tensor] = []
    for seed in seed_list:
        seed_embedding_root = embedding_root / variant_name / backbone / f"seed{seed}"
        payloads = {
            split_name: _load_embedding_payload(seed_embedding_root / f"annotated_sidecar_{split_name}_embeddings.pt")
            for split_name in ("train", "val", "test")
        }
        split_features: dict[str, torch.Tensor] = {}
        for split_name in ("train", "val", "test"):
            probe_id_to_index = {
                str(probe_id): index
                for index, probe_id in enumerate(payloads[split_name]["probe_id"])
            }
            ordered_indices = torch.tensor(
                [probe_id_to_index[probe_id] for probe_id in probe_ids_by_split[split_name]],
                dtype=torch.long,
            )
            split_features[split_name] = payloads[split_name][pool].index_select(0, ordered_indices).float()

        train_features, val_features, test_features = _standardize_features(
            split_features["train"],
            split_features["val"],
            split_features["test"],
        )
        train_limit = None if probe_model == "linear" else 100000
        model, _ = _train_text_probe(
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
        test_text = tfidf_by_split["test"].index_select(0, ordered_common_indices)
        test_pred = predicted_test.index_select(0, ordered_common_indices)
        board_rr_by_seed.append(_strict_reciprocal_ranks(test_pred, test_text))
        text_rr_by_seed.append(_strict_reciprocal_ranks(test_text, test_pred))

    return {
        "strict_board_to_text_rr": torch.stack(board_rr_by_seed, dim=0).mean(dim=0),
        "strict_text_to_board_rr": torch.stack(text_rr_by_seed, dim=0).mean(dim=0),
        "common_query_count": len(common_test_ids),
    }


def compare_retrieval_variants(
    *,
    results_root: str | Path = "outputs/week17/comment_retrieval_v6",
    embedding_root: str | Path = "outputs/week17/embedding_exports",
    output_dir: str | Path = "outputs/week18/retrieval_comparison",
    baseline_variant: str = "current_mixed_baseline",
    comparison_variants: list[str] | None = None,
    backbone_seeds: list[int] | None = None,
    config: RetrievalComparisonConfig | None = None,
) -> dict[str, Any]:
    comparison_config = config or RetrievalComparisonConfig()
    seed_list = backbone_seeds or [11, 17, 23]
    results_path = Path(results_root)
    embedding_path = Path(embedding_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    baseline_payload = json.loads(
        (results_path / baseline_variant / "comment_retrieval_results.json").read_text(encoding="utf-8")
    )
    baseline_test_rows = _load_jsonl(results_path / baseline_variant / "probe_subset" / "annotated_sidecar_test.jsonl")
    variants = comparison_variants or ["family_balanced", "family_balanced_plus_style_normalized"]

    comparisons: list[dict[str, Any]] = []
    markdown_lines = ["# Retrieval Variant Comparison", ""]
    for comparison_variant in variants:
        comparison_payload = json.loads(
            (results_path / comparison_variant / "comment_retrieval_results.json").read_text(encoding="utf-8")
        )
        comparison_test_rows = _load_jsonl(
            results_path / comparison_variant / "probe_subset" / "annotated_sidecar_test.jsonl"
        )
        common_test_ids = _common_test_probe_ids(baseline_test_rows, comparison_test_rows)
        if not common_test_ids:
            continue

        baseline_board_best = _best_row(baseline_payload["aggregate"], "strict_board_to_text_mrr_mean")
        baseline_text_best = _best_row(baseline_payload["aggregate"], "strict_text_to_board_mrr_mean")
        comparison_board_best = _best_row(comparison_payload["aggregate"], "strict_board_to_text_mrr_mean")
        comparison_text_best = _best_row(comparison_payload["aggregate"], "strict_text_to_board_mrr_mean")

        score_specs = (
            (
                "board_to_text",
                baseline_board_best,
                comparison_board_best,
                "strict_board_to_text_rr",
            ),
            (
                "text_to_board",
                baseline_text_best,
                comparison_text_best,
                "strict_text_to_board_rr",
            ),
        )
        markdown_lines.extend([f"## {baseline_variant} vs {comparison_variant}", ""])
        for direction, baseline_best, candidate_best, score_key in score_specs:
            baseline_scores = _variant_query_scores(
                variant_name=baseline_variant,
                results_root=results_path,
                embedding_root=embedding_path,
                backbone=str(baseline_best["backbone"]),
                pool=str(baseline_best["pool"]),
                probe_model=str(baseline_best["probe_model"]),
                common_test_ids=common_test_ids,
                seed_list=seed_list,
                config=comparison_config,
            )[score_key]
            candidate_scores = _variant_query_scores(
                variant_name=comparison_variant,
                results_root=results_path,
                embedding_root=embedding_path,
                backbone=str(candidate_best["backbone"]),
                pool=str(candidate_best["pool"]),
                probe_model=str(candidate_best["probe_model"]),
                common_test_ids=common_test_ids,
                seed_list=seed_list,
                config=comparison_config,
            )[score_key]
            deltas = candidate_scores - baseline_scores
            ci_low, ci_high = _bootstrap_delta_ci(
                deltas,
                samples=comparison_config.bootstrap_samples,
                seed=comparison_config.bootstrap_seed,
            )
            positive = int((deltas > 0).sum().item())
            negative = int((deltas < 0).sum().item())
            ties = int((deltas == 0).sum().item())
            row = {
                "baseline_variant": baseline_variant,
                "comparison_variant": comparison_variant,
                "direction": direction,
                "common_query_count": len(common_test_ids),
                "baseline_backbone": baseline_best["backbone"],
                "baseline_pool": baseline_best["pool"],
                "baseline_probe_model": baseline_best["probe_model"],
                "comparison_backbone": candidate_best["backbone"],
                "comparison_pool": candidate_best["pool"],
                "comparison_probe_model": candidate_best["probe_model"],
                "baseline_mean_strict_mrr_on_shared_queries": float(baseline_scores.mean().item()),
                "comparison_mean_strict_mrr_on_shared_queries": float(candidate_scores.mean().item()),
                "delta_mean_strict_mrr": float(deltas.mean().item()),
                "delta_bootstrap_ci_low": ci_low,
                "delta_bootstrap_ci_high": ci_high,
                "positive_query_count": positive,
                "negative_query_count": negative,
                "tie_query_count": ties,
                "approx_sign_test_pvalue": _approx_two_sided_sign_pvalue(positive, negative),
                "bootstrap_interpretation": _ci_interpretation(ci_low, ci_high),
            }
            comparisons.append(row)
            markdown_lines.append(
                f"- `{direction}`: shared_queries=`{row['common_query_count']}`, "
                f"baseline=`{row['baseline_mean_strict_mrr_on_shared_queries']:.6f}`, "
                f"candidate=`{row['comparison_mean_strict_mrr_on_shared_queries']:.6f}`, "
                f"delta=`{row['delta_mean_strict_mrr']:.6f}`, "
                f"bootstrap_CI=`[{row['delta_bootstrap_ci_low']:.6f}, {row['delta_bootstrap_ci_high']:.6f}]`, "
                f"sign_p~`{row['approx_sign_test_pvalue']:.4f}`, "
                f"interpretation=`{row['bootstrap_interpretation']}`"
            )
        markdown_lines.append("")

    payload = {
        "config": asdict(comparison_config),
        "comparisons": comparisons,
    }
    json_path = output_path / "comparison_results.json"
    md_path = output_path / "comparison_results.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "payload": payload,
    }
