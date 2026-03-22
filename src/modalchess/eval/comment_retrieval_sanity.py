"""Sanity audit for annotated-sidecar comment retrieval."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch

from modalchess.data.comment_retrieval_eval import (
    CommentRetrievalEvalConfig,
    build_comment_retrieval_eval_regime,
)
from modalchess.eval.embedding_export import EmbeddingExportConfig, export_embeddings_for_checkpoint
from modalchess.eval.raw_text_retrieval import (
    _align_rows_by_probe_id,
    _build_vocab,
    _chunked_retrieval_metrics,
    _documents_for_family,
    _load_embedding_payload,
    _load_jsonl,
    _normalize_rows,
    _standardize_features,
    _tfidf_matrix,
    _train_text_probe,
    run_raw_text_retrieval_probes,
)


@dataclass(slots=True)
class CommentRetrievalSanityConfig:
    alternate_salt: str = "modalchess_week11_comment_eval_alt"
    larger_subset_salt: str = "modalchess_week11_comment_eval_large"
    train_limit: int = 100000
    val_limit: int = 5000
    test_limit: int = 5000
    larger_val_limit: int = 10000
    larger_test_limit: int = 10000
    mate_min_df: int = 25
    max_vocab_size: int = 512
    embedding_batch_size: int = 1024


def compute_rank_diagnostics(
    score_matrix: torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> dict[str, float]:
    target_count = score_matrix.size(0)
    diagonal_indices = torch.arange(target_count, dtype=torch.long)
    target_scores = score_matrix[diagonal_indices, diagonal_indices]
    standard_ranks = (score_matrix > target_scores.unsqueeze(1)).sum(dim=1) + 1
    tie_mask = torch.isclose(score_matrix, target_scores.unsqueeze(1), atol=atol, rtol=rtol)
    strict_ranks = (score_matrix > target_scores.unsqueeze(1)).sum(dim=1) + tie_mask.sum(dim=1)
    tied_competitors = tie_mask.sum(dim=1) - 1

    def _metrics(ranks: torch.Tensor) -> dict[str, float]:
        return {
            "recall_at_1": float((ranks <= 1).float().mean().item()),
            "recall_at_5": float((ranks <= 5).float().mean().item()),
            "mrr": float((1.0 / ranks.float()).mean().item()),
        }

    standard_metrics = _metrics(standard_ranks)
    strict_metrics = _metrics(strict_ranks)
    return {
        "standard_recall_at_1": standard_metrics["recall_at_1"],
        "standard_recall_at_5": standard_metrics["recall_at_5"],
        "standard_mrr": standard_metrics["mrr"],
        "strict_recall_at_1": strict_metrics["recall_at_1"],
        "strict_recall_at_5": strict_metrics["recall_at_5"],
        "strict_mrr": strict_metrics["mrr"],
        "rank_1_count": int((standard_ranks == 1).sum().item()),
        "rank_2_to_5_count": int(((standard_ranks >= 2) & (standard_ranks <= 5)).sum().item()),
        "rank_over_5_count": int((standard_ranks > 5).sum().item()),
        "queries_with_tied_competitors": int((tied_competitors > 0).sum().item()),
        "mean_tied_competitor_count": float(tied_competitors.float().mean().item()),
        "max_tied_competitor_count": int(tied_competitors.max().item()) if tied_competitors.numel() else 0,
        "strict_rank_penalty_count": int((strict_ranks > standard_ranks).sum().item()),
    }


def _checkpoint_path(backbone: str, seed: int) -> Path:
    if backbone == "g1":
        return Path(f"outputs/week3/exp3_ground_state/seed{seed}/best_model.pt")
    if backbone == "g3":
        return Path(f"outputs/week3/exp3_ground_state_legality/seed{seed}/best_model.pt")
    raise ValueError(f"unsupported backbone: {backbone}")


def _ensure_embeddings(
    *,
    subset_root: Path,
    embedding_root: Path,
    seeds: list[int],
    batch_size: int,
) -> None:
    dataset_paths = {
        "annotated_sidecar_train": subset_root / "annotated_sidecar_train.jsonl",
        "annotated_sidecar_val": subset_root / "annotated_sidecar_val.jsonl",
        "annotated_sidecar_test": subset_root / "annotated_sidecar_test.jsonl",
    }
    for backbone in ("g1", "g3"):
        for seed in seeds:
            destination = embedding_root / backbone / f"seed{seed}"
            if (destination / "annotated_sidecar_test_embeddings.pt").exists():
                continue
            export_embeddings_for_checkpoint(
                checkpoint_path=_checkpoint_path(backbone, seed),
                dataset_paths=dataset_paths,
                output_dir=destination,
                config=EmbeddingExportConfig(batch_size=batch_size, output_format="pt"),
            )


def _run_regime(
    *,
    subset_root: Path,
    embedding_root: Path,
    output_root: Path,
    seeds: list[int],
    config: CommentRetrievalSanityConfig,
    prefix: str,
) -> dict[str, Any]:
    return run_raw_text_retrieval_probes(
        embedding_root=embedding_root,
        corpus_root=subset_root,
        output_dir=output_root,
        backbone_seeds=seeds,
        mate_min_df=config.mate_min_df,
        max_vocab_size=config.max_vocab_size,
        families=["annotated_sidecar"],
        output_prefix=prefix,
    )


def _best_rows_by_backbone(aggregate_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    winners: dict[str, dict[str, Any]] = {}
    for backbone in ("g1", "g3"):
        rows = [row for row in aggregate_rows if str(row["backbone"]) == backbone]
        if rows:
            winners[backbone] = max(rows, key=lambda row: float(row["board_to_text_mrr_mean"]))
    return winners


def _prepare_probe(
    *,
    corpus_root: Path,
    embedding_root: Path,
    backbone: str,
    seed: int,
    pool: str,
    probe_model: str,
    config: CommentRetrievalSanityConfig,
) -> dict[str, Any]:
    corpus_rows_by_split = {
        split_name: _load_jsonl(corpus_root / f"annotated_sidecar_{split_name}.jsonl")
        for split_name in ("train", "val", "test")
    }
    aligned_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split_name, rows in corpus_rows_by_split.items():
        aligned_rows, _ = _align_rows_by_probe_id(rows, None)
        aligned_rows_by_split[split_name] = aligned_rows
    documents_by_split = {
        split_name: _documents_for_family("annotated_sidecar", aligned_rows_by_split[split_name], None)
        for split_name in ("train", "val", "test")
    }
    vocab, token_to_index, idf = _build_vocab(
        documents_by_split["train"],
        min_df=config.mate_min_df,
        max_vocab_size=config.max_vocab_size,
    )
    tfidf_by_split = {
        split_name: _normalize_rows(_tfidf_matrix(documents_by_split[split_name], token_to_index, idf))
        for split_name in ("train", "val", "test")
    }
    probe_ids_by_split = {
        split_name: [str(row["probe_id"]) for row in aligned_rows_by_split[split_name]]
        for split_name in ("train", "val", "test")
    }
    payload_root = embedding_root / backbone / f"seed{seed}"
    embedding_payloads = {
        split_name: _load_embedding_payload(payload_root / f"annotated_sidecar_{split_name}_embeddings.pt")
        for split_name in ("train", "val", "test")
    }
    split_features: dict[str, torch.Tensor] = {}
    for split_name in ("train", "val", "test"):
        payload = embedding_payloads[split_name]
        probe_id_to_index = {str(probe_id): index for index, probe_id in enumerate(payload["probe_id"])}
        ordered_indices = torch.tensor(
            [probe_id_to_index[probe_id] for probe_id in probe_ids_by_split[split_name]],
            dtype=torch.long,
        )
        split_features[split_name] = payload[pool].index_select(0, ordered_indices).float()
    train_features, val_features, test_features = _standardize_features(
        split_features["train"],
        split_features["val"],
        split_features["test"],
    )
    model, val_alignment = _train_text_probe(
        model_kind=probe_model,
        train_features=train_features,
        train_targets=tfidf_by_split["train"],
        val_features=val_features,
        val_targets=tfidf_by_split["val"],
        seed=seed,
        max_train_rows=100000 if probe_model == "mlp" else None,
    )
    model.eval()
    with torch.no_grad():
        predicted_test = _normalize_rows(model(test_features))
    score_matrix = predicted_test @ tfidf_by_split["test"].transpose(0, 1)
    diagnostics = compute_rank_diagnostics(score_matrix)
    standard = _chunked_retrieval_metrics(predicted_test, tfidf_by_split["test"])
    comment_counter = Counter(documents_by_split["test"])
    duplicate_rows = sum(count - 1 for count in comment_counter.values() if count > 1)
    return {
        "val_alignment": val_alignment,
        "standard_metrics": {
            "recall_at_1": standard[0],
            "recall_at_5": standard[1],
            "mrr": standard[2],
        },
        "rank_diagnostics": diagnostics,
        "test_row_count": len(documents_by_split["test"]),
        "unique_comment_text_count": len(comment_counter),
        "duplicate_comment_rows": duplicate_rows,
        "duplicate_comment_rate": (duplicate_rows / len(documents_by_split["test"])) if documents_by_split["test"] else 0.0,
    }


def _regime_summary(name: str, result: dict[str, Any]) -> dict[str, Any]:
    aggregate_rows = result["aggregate"]
    return {
        "name": name,
        "best_board_to_text": max(aggregate_rows, key=lambda row: float(row["board_to_text_mrr_mean"])),
        "best_text_to_board": max(aggregate_rows, key=lambda row: float(row["text_to_board_mrr_mean"])),
        "best_by_backbone": _best_rows_by_backbone(aggregate_rows),
    }


def audit_comment_retrieval_sanity(
    *,
    repo_root: str | Path = ".",
    output_dir: str | Path = "outputs/week11/comment_retrieval_sanity",
    config: CommentRetrievalSanityConfig | None = None,
) -> dict[str, Any]:
    sanity_config = config or CommentRetrievalSanityConfig()
    root = Path(repo_root)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [11, 17, 23]

    baseline_result = json.loads(
        (root / "outputs/week10/comment_retrieval/comment_retrieval_results.json").read_text(encoding="utf-8")
    )
    baseline_summary = _regime_summary("week10_baseline", baseline_result)

    alternate_subset_root = output_root / "alternate_subset" / "probe_subset"
    build_comment_retrieval_eval_regime(
        input_root=root / "data/pilot/annotated_sidecar_v1",
        output_root=alternate_subset_root.parent,
        config=CommentRetrievalEvalConfig(
            train_limit=sanity_config.train_limit,
            val_limit=sanity_config.val_limit,
            test_limit=sanity_config.test_limit,
            salt=sanity_config.alternate_salt,
            require_non_empty_comment=True,
            stratify_by="comment_source",
        ),
    )
    larger_subset_root = output_root / "larger_subset" / "probe_subset"
    build_comment_retrieval_eval_regime(
        input_root=root / "data/pilot/annotated_sidecar_v1",
        output_root=larger_subset_root.parent,
        config=CommentRetrievalEvalConfig(
            train_limit=sanity_config.train_limit,
            val_limit=sanity_config.larger_val_limit,
            test_limit=sanity_config.larger_test_limit,
            salt=sanity_config.larger_subset_salt,
            require_non_empty_comment=True,
            stratify_by="comment_source",
        ),
    )

    alternate_embedding_root = output_root / "alternate_subset" / "embedding_exports"
    larger_embedding_root = output_root / "larger_subset" / "embedding_exports"
    _ensure_embeddings(
        subset_root=alternate_subset_root,
        embedding_root=alternate_embedding_root,
        seeds=seeds,
        batch_size=sanity_config.embedding_batch_size,
    )
    _ensure_embeddings(
        subset_root=larger_subset_root,
        embedding_root=larger_embedding_root,
        seeds=seeds,
        batch_size=sanity_config.embedding_batch_size,
    )

    alternate_result = _run_regime(
        subset_root=alternate_subset_root,
        embedding_root=alternate_embedding_root,
        output_root=output_root / "alternate_subset",
        seeds=seeds,
        config=sanity_config,
        prefix="alternate_comment_retrieval",
    )
    larger_result = _run_regime(
        subset_root=larger_subset_root,
        embedding_root=larger_embedding_root,
        output_root=output_root / "larger_subset",
        seeds=seeds,
        config=sanity_config,
        prefix="larger_comment_retrieval",
    )

    alternate_summary = _regime_summary("alternate_subset", alternate_result)
    larger_summary = _regime_summary("larger_subset", larger_result)
    best_board = baseline_summary["best_board_to_text"]
    baseline_probe_diagnostics = _prepare_probe(
        corpus_root=root / "outputs/week10/comment_retrieval/probe_subset",
        embedding_root=root / "outputs/week10/embedding_exports",
        backbone=str(best_board["backbone"]),
        seed=11,
        pool=str(best_board["pool"]),
        probe_model=str(best_board["probe_model"]),
        config=sanity_config,
    )

    baseline_random_r1 = 1.0 / max(int(best_board["test_rows_mean"]), 1)
    larger_random_r1 = 1.0 / max(int(larger_summary["best_board_to_text"]["test_rows_mean"]), 1)
    board_mrr_values = [
        float(baseline_summary["best_board_to_text"]["board_to_text_mrr_mean"]),
        float(alternate_summary["best_board_to_text"]["board_to_text_mrr_mean"]),
        float(larger_summary["best_board_to_text"]["board_to_text_mrr_mean"]),
    ]
    stability = {
        "winner_by_regime": {
            "baseline": str(baseline_summary["best_board_to_text"]["backbone"]),
            "alternate_subset": str(alternate_summary["best_board_to_text"]["backbone"]),
            "larger_subset": str(larger_summary["best_board_to_text"]["backbone"]),
        },
        "board_to_text_mrr_range": max(board_mrr_values) - min(board_mrr_values),
        "alternate_minus_baseline_board_to_text_mrr": (
            float(alternate_summary["best_board_to_text"]["board_to_text_mrr_mean"])
            - float(baseline_summary["best_board_to_text"]["board_to_text_mrr_mean"])
        ),
        "larger_minus_baseline_board_to_text_mrr": (
            float(larger_summary["best_board_to_text"]["board_to_text_mrr_mean"])
            - float(baseline_summary["best_board_to_text"]["board_to_text_mrr_mean"])
        ),
    }
    clearly_above_trivial = (
        float(baseline_summary["best_board_to_text"]["board_to_text_recall_at_1_mean"]) > (baseline_random_r1 * 100.0)
        and float(alternate_summary["best_board_to_text"]["board_to_text_recall_at_1_mean"]) > (baseline_random_r1 * 100.0)
        and float(larger_summary["best_board_to_text"]["board_to_text_recall_at_1_mean"]) > (larger_random_r1 * 100.0)
    )
    duplicate_or_tie_artifact_dominates = (
        baseline_probe_diagnostics["rank_diagnostics"]["standard_recall_at_1"]
        - baseline_probe_diagnostics["rank_diagnostics"]["strict_recall_at_1"]
        > 0.05
        and baseline_probe_diagnostics["rank_diagnostics"]["queries_with_tied_competitors"]
        > int(baseline_probe_diagnostics["test_row_count"] * 0.25)
    )
    regime_stable = stability["board_to_text_mrr_range"] <= 0.01
    audit = {
        "config": asdict(sanity_config),
        "baseline": baseline_summary,
        "alternate_subset": alternate_summary,
        "larger_subset": larger_summary,
        "baseline_probe_diagnostics": baseline_probe_diagnostics,
        "stability": stability,
        "clearly_above_trivial_baseline": clearly_above_trivial,
        "duplicate_or_tie_artifact_dominates": duplicate_or_tie_artifact_dominates,
        "subset_regime_stable": regime_stable,
        "exact_vs_approximate_retrieval": {
            "path": "exact",
            "notes": "raw_text_retrieval uses exact dense matrix similarity; no ANN path is used.",
        },
        "gate_recommendation": (
            "PASS"
            if clearly_above_trivial and regime_stable and not duplicate_or_tie_artifact_dominates
            else "HOLD"
        ),
    }

    json_path = output_root / "comment_retrieval_sanity.json"
    md_path = output_root / "comment_retrieval_sanity.md"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = ["# Comment Retrieval Sanity Audit", ""]
    lines.append(f"- gate_recommendation: `{audit['gate_recommendation']}`")
    lines.append(f"- clearly_above_trivial_baseline: {clearly_above_trivial}")
    lines.append(f"- subset_regime_stable: {regime_stable}")
    lines.append(f"- duplicate_or_tie_artifact_dominates: {duplicate_or_tie_artifact_dominates}")
    lines.append("")
    lines.append("## Baseline Duplicate/Tie Diagnostics")
    lines.append(
        f"- duplicate_comment_rate: {baseline_probe_diagnostics['duplicate_comment_rate']:.6f} "
        f"({baseline_probe_diagnostics['duplicate_comment_rows']} duplicate rows)"
    )
    lines.append(
        f"- standard R@1/R@5/MRR: {baseline_probe_diagnostics['standard_metrics']['recall_at_1']:.4f} / "
        f"{baseline_probe_diagnostics['standard_metrics']['recall_at_5']:.4f} / "
        f"{baseline_probe_diagnostics['standard_metrics']['mrr']:.4f}"
    )
    lines.append(
        f"- strict   R@1/R@5/MRR: {baseline_probe_diagnostics['rank_diagnostics']['strict_recall_at_1']:.4f} / "
        f"{baseline_probe_diagnostics['rank_diagnostics']['strict_recall_at_5']:.4f} / "
        f"{baseline_probe_diagnostics['rank_diagnostics']['strict_mrr']:.4f}"
    )
    lines.append(
        f"- tied competitors: {baseline_probe_diagnostics['rank_diagnostics']['queries_with_tied_competitors']} "
        f"queries, mean={baseline_probe_diagnostics['rank_diagnostics']['mean_tied_competitor_count']:.2f}, "
        f"max={baseline_probe_diagnostics['rank_diagnostics']['max_tied_competitor_count']}"
    )
    lines.append(
        f"- rank histogram: r1={baseline_probe_diagnostics['rank_diagnostics']['rank_1_count']}, "
        f"r2_5={baseline_probe_diagnostics['rank_diagnostics']['rank_2_to_5_count']}, "
        f"r>5={baseline_probe_diagnostics['rank_diagnostics']['rank_over_5_count']}"
    )
    lines.append("")
    lines.append("## Regime Stability")
    for regime_name, summary in (
        ("baseline", baseline_summary),
        ("alternate_subset", alternate_summary),
        ("larger_subset", larger_summary),
    ):
        best = summary["best_board_to_text"]
        lines.append(
            f"- `{regime_name}` best board_to_text: backbone={best['backbone']}, pool={best['pool']}, "
            f"probe={best['probe_model']}, R@1={float(best['board_to_text_recall_at_1_mean']):.4f}, "
            f"MRR={float(best['board_to_text_mrr_mean']):.4f}"
        )
    lines.append(f"- board_to_text_mrr_range: {stability['board_to_text_mrr_range']:.6f}")
    lines.append(
        f"- winner_by_regime: baseline={stability['winner_by_regime']['baseline']}, "
        f"alternate={stability['winner_by_regime']['alternate_subset']}, "
        f"larger={stability['winner_by_regime']['larger_subset']}"
    )
    lines.append("")
    lines.append("## Retrieval Path")
    lines.append("- exact dense similarity matrix; no ANN or approximate index was used")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit["json_path"] = str(json_path)
    audit["md_path"] = str(md_path)
    return audit
