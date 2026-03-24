"""Week-18 readiness falsification analyses for balanced retrieval variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import statistics
from pathlib import Path
import subprocess
from typing import Any

import torch
import yaml

from modalchess.eval.eval_baseline import run_evaluation
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
from modalchess.eval.retrieval_comparison import (
    _approx_two_sided_sign_pvalue,
    _bootstrap_delta_ci,
    _ci_interpretation,
    _strict_reciprocal_ranks,
)


POLICY_EXPERIMENTS = {
    "g1": "exp3_ground_state",
    "g3": "exp3_ground_state_legality",
}

GROUNDING_METRIC_KEYS = (
    "occupied_square_accuracy",
    "piece_macro_f1",
    "legality_average_precision",
)


@dataclass(slots=True)
class PoolThresholds:
    primary_min_test_rows: int = 500
    strong_primary_min_test_rows: int = 1000
    exploratory_max_test_rows: int = 199


@dataclass(slots=True)
class ReadinessFalsificationConfig:
    holdout_manifest_path: str = "data/pilot/annotated_sidecar_holdout_v2/manifests/holdout_manifest.yaml"
    week17_results_root: str = "outputs/week17/comment_retrieval_v6"
    week18_holdout_root: str = "outputs/week18/source_holdout_balanced"
    week3_root: str = "outputs/week3"
    checkpoint_eval_root: str = "outputs/week18/checkpoint_honesty"
    output_root: str = "outputs/week18"
    baseline_variant: str = "current_mixed_baseline"
    comparison_variants: tuple[str, ...] = (
        "family_balanced",
        "family_balanced_plus_style_normalized",
    )
    backbone_seeds: tuple[int, ...] = (11, 17, 23)
    mate_min_df: int = 25
    max_vocab_size: int = 512
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 1234
    null_seed: int = 20260324
    thresholds: PoolThresholds = field(default_factory=PoolThresholds)


def classify_shared_pool(shared_test_rows: int, thresholds: PoolThresholds | None = None) -> str:
    """Classify a shared regime by test-pool size."""
    active = thresholds or PoolThresholds()
    if shared_test_rows <= active.exploratory_max_test_rows:
        return "exploratory_only"
    if shared_test_rows < active.primary_min_test_rows:
        return "secondary_shared"
    if shared_test_rows >= active.strong_primary_min_test_rows:
        return "strong_primary"
    return "primary_shared"


def build_family_deranged_permutation(
    labels: list[str],
    *,
    seed: int,
    max_attempts: int = 2048,
) -> torch.Tensor:
    """Return a deterministic permutation where no row keeps the same family label."""
    total = len(labels)
    if total == 0:
        return torch.empty(0, dtype=torch.long)
    counts: dict[str, int] = {}
    grouped_indices: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        counts[label] = counts.get(label, 0) + 1
        grouped_indices.setdefault(label, []).append(index)
    if max(counts.values(), default=0) > total // 2:
        raise ValueError("family derangement가 불가능하다: 하나의 family가 절반을 초과한다.")
    ordered_indices: list[int] = []
    for label in sorted(grouped_indices):
        ordered_indices.extend(grouped_indices[label])
    shift = max(counts.values(), default=0)
    rotated = ordered_indices[shift:] + ordered_indices[:shift]
    permutation = torch.empty(total, dtype=torch.long)
    for source_index, target_index in zip(ordered_indices, rotated, strict=True):
        permutation[source_index] = target_index
    if torch.any(permutation == torch.arange(total, dtype=torch.long)):
        raise ValueError("family derangement rotation이 identity를 만들었다.")
    if any(labels[int(permutation[index])] == labels[index] for index in range(total)):
        raise ValueError("family derangement rotation이 같은 family를 남겼다.")
    return permutation


def selection_epochs_from_epoch_metrics(epoch_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute policy-best and grounding-best epochs from train metrics."""
    if not epoch_metrics:
        raise ValueError("epoch metrics가 비어 있다.")

    def _grounding_score(metric_row: dict[str, Any]) -> float:
        val_metrics = dict(metric_row.get("val") or {})
        missing = [key for key in GROUNDING_METRIC_KEYS if key not in val_metrics]
        if missing:
            raise ValueError(f"grounding score 계산에 필요한 metric이 없다: {missing}")
        return statistics.fmean(float(val_metrics[key]) for key in GROUNDING_METRIC_KEYS)

    policy_best = min(
        epoch_metrics,
        key=lambda row: (float((row.get("val") or {})["target_move_nll"]), int(row["epoch"])),
    )
    grounding_best = max(
        epoch_metrics,
        key=lambda row: (_grounding_score(row), -int(row["epoch"])),
    )
    return {
        "policy_best_epoch": int(policy_best["epoch"]),
        "grounding_best_epoch": int(grounding_best["epoch"]),
        "policy_best_metric": float((policy_best.get("val") or {})["target_move_nll"]),
        "grounding_best_metric": _grounding_score(grounding_best),
        "epochs_match": int(policy_best["epoch"]) == int(grounding_best["epoch"]),
    }


def _json_load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_dump(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _markdown_write(path: str | Path, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def _git_commit_hash(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _holdout_manifest(config: ReadinessFalsificationConfig) -> dict[str, Any]:
    return yaml.safe_load(Path(config.holdout_manifest_path).read_text(encoding="utf-8")) or {}


def _lookup_regime_name(
    config: ReadinessFalsificationConfig,
    *,
    category: str,
    holdout_field: str | None,
    holdout_value: str | None,
) -> str:
    payload = _variant_results_payload(config, config.baseline_variant)
    matches = [
        dict(row)
        for row in payload.get("results") or []
        if str(row.get("category")) == category
        and str(row.get("holdout_field")) == str(holdout_field)
        and str(row.get("holdout_value")) == str(holdout_value)
    ]
    if not matches:
        if category == "mixed_baseline":
            return "mixed_baseline"
        raise ValueError(
            f"regime_name을 찾지 못했다: category={category}, holdout_field={holdout_field}, holdout_value={holdout_value}"
        )
    return str(matches[0]["regime_name"])


def _shared_regimes_with_buckets(config: ReadinessFalsificationConfig) -> list[dict[str, Any]]:
    manifest = _holdout_manifest(config)
    regimes: list[dict[str, Any]] = []
    for regime in manifest.get("shared_regimes", []):
        enriched = dict(regime)
        shared_rows = int(enriched.get("shared_test_rows") or 0)
        enriched["regime_name"] = _lookup_regime_name(
            config,
            category=str(enriched.get("category")),
            holdout_field=(None if enriched.get("holdout_field") is None else str(enriched.get("holdout_field"))),
            holdout_value=(None if enriched.get("holdout_value") is None else str(enriched.get("holdout_value"))),
        )
        enriched["pool_bucket"] = classify_shared_pool(shared_rows, config.thresholds)
        enriched["is_primary"] = shared_rows >= config.thresholds.primary_min_test_rows
        enriched["is_strong_primary"] = shared_rows >= config.thresholds.strong_primary_min_test_rows
        enriched["is_large_source_holdout"] = (
            enriched["category"] != "mixed_baseline"
            and shared_rows >= config.thresholds.primary_min_test_rows
        )
        regimes.append(enriched)
    return regimes


def _variant_results_payload(config: ReadinessFalsificationConfig, variant_name: str) -> dict[str, Any]:
    root = Path(config.week18_holdout_root) / "per_variant" / variant_name / "results.json"
    return _json_load(root)


def _variant_regime_aggregate_rows(payload: dict[str, Any], regime_name: str) -> list[dict[str, Any]]:
    aggregate = payload.get("aggregate") or []
    rows = [dict(row) for row in aggregate if str(row.get("regime_name")) == regime_name]
    if not rows:
        raise ValueError(f"{regime_name}에 대한 aggregate rows가 없다.")
    return rows


def _best_aggregate_row(rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row[metric_name]),
            -float(row.get("test_rows_mean") or 0.0),
            str(row["backbone"]),
            str(row["pool"]),
            str(row["probe_model"]),
        ),
    )


def _best_choice_for_variant_regime(
    config: ReadinessFalsificationConfig,
    *,
    variant_name: str,
    regime_name: str,
    direction: str,
) -> dict[str, Any]:
    payload = _variant_results_payload(config, variant_name)
    rows = _variant_regime_aggregate_rows(payload, regime_name)
    metric_name = (
        "strict_board_to_text_mrr_mean"
        if direction == "board_to_text"
        else "strict_text_to_board_mrr_mean"
    )
    best_row = _best_aggregate_row(rows, metric_name)
    return {
        "variant": variant_name,
        "regime_name": regime_name,
        "direction": direction,
        "metric_name": metric_name,
        "backbone": str(best_row["backbone"]),
        "pool": str(best_row["pool"]),
        "probe_model": str(best_row["probe_model"]),
        "metric_value": float(best_row[metric_name]),
        "test_rows_mean": float(best_row["test_rows_mean"]),
    }


def _prepare_regime_variant_inputs(
    config: ReadinessFalsificationConfig,
    *,
    variant_name: str,
    regime_name: str,
) -> dict[str, Any]:
    payload = _variant_results_payload(config, variant_name)
    result_rows = [dict(row) for row in payload.get("results") or [] if str(row.get("regime_name")) == regime_name]
    if not result_rows:
        raise ValueError(f"{variant_name}/{regime_name} result rows가 없다.")
    regime_dir = Path(str(result_rows[0]["regime_dir"]))
    split_rows = {
        split_name: _load_jsonl(regime_dir / f"annotated_sidecar_{split_name}.jsonl")
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
        raise ValueError(f"{variant_name}/{regime_name} vocabulary가 비어 있다.")
    tfidf_by_split = {
        split_name: _normalize_rows(_tfidf_matrix(documents_by_split[split_name], token_to_index, idf))
        for split_name in ("train", "val", "test")
    }
    probe_ids_by_split = {
        split_name: [str(row["probe_id"]) for row in split_rows[split_name]]
        for split_name in ("train", "val", "test")
    }
    return {
        "regime_dir": regime_dir,
        "split_rows": split_rows,
        "tfidf_by_split": tfidf_by_split,
        "probe_ids_by_split": probe_ids_by_split,
        "embedding_cache_dir": (
            Path(config.week18_holdout_root)
            / "per_variant"
            / variant_name
            / "embedding_cache"
            / regime_name
        ),
    }


def _deterministic_train_probe(
    *,
    model_kind: str,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    seed: int,
) -> tuple[torch.nn.Module, float]:
    previous_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        return _train_text_probe(
            model_kind=model_kind,
            train_features=train_features,
            train_targets=train_targets,
            val_features=val_features,
            val_targets=val_targets,
            seed=seed,
            max_train_rows=None if model_kind == "linear" else 100000,
        )
    finally:
        torch.random.set_rng_state(previous_state)


def _shared_test_ids(shared_ids_path: str | Path) -> list[str]:
    payload = _json_load(shared_ids_path)
    return [str(item) for item in payload.get("shared_test_probe_ids") or []]


def _assignment_indices(
    *,
    control: str,
    rows: list[dict[str, Any]],
    seed: int,
) -> torch.Tensor:
    total = len(rows)
    if control == "real":
        return torch.arange(total, dtype=torch.long)
    if control == "text_shuffle_within_pool":
        return torch.randperm(total, generator=torch.Generator().manual_seed(seed))
    if control == "source_family_permutation":
        labels = [str(row.get("source_family") or "unknown") for row in rows]
        return build_family_deranged_permutation(labels, seed=seed)
    raise ValueError(f"지원하지 않는 null control: {control}")


def _query_scores_for_choice(
    config: ReadinessFalsificationConfig,
    *,
    prepared: dict[str, Any],
    choice: dict[str, Any],
    shared_probe_ids: list[str],
    control: str,
    control_seed: int,
) -> dict[str, Any]:
    probe_ids = list(prepared["probe_ids_by_split"]["test"])
    index_by_probe_id = {probe_id: index for index, probe_id in enumerate(probe_ids)}
    common_indices = torch.tensor(
        [index_by_probe_id[probe_id] for probe_id in shared_probe_ids],
        dtype=torch.long,
    )
    common_rows = [prepared["split_rows"]["test"][int(index)] for index in common_indices.tolist()]
    assignment = _assignment_indices(control=control, rows=common_rows, seed=control_seed)

    board_rr_by_seed: list[torch.Tensor] = []
    text_rr_by_seed: list[torch.Tensor] = []
    val_alignment_by_seed: list[float] = []
    for seed in config.backbone_seeds:
        embedding_dir = (
            Path(prepared["embedding_cache_dir"])
            / str(choice["backbone"])
            / f"seed{seed}"
        )
        payloads = {
            split_name: _load_embedding_payload(embedding_dir / f"annotated_sidecar_{split_name}_embeddings.pt")
            for split_name in ("train", "val", "test")
        }
        split_features: dict[str, torch.Tensor] = {}
        for split_name in ("train", "val", "test"):
            probe_id_to_index = {
                str(probe_id): index
                for index, probe_id in enumerate(payloads[split_name]["probe_id"])
            }
            ordered_indices = torch.tensor(
                [probe_id_to_index[probe_id] for probe_id in prepared["probe_ids_by_split"][split_name]],
                dtype=torch.long,
            )
            split_features[split_name] = payloads[split_name][str(choice["pool"])].index_select(0, ordered_indices).float()

        train_features, val_features, test_features = _standardize_features(
            split_features["train"],
            split_features["val"],
            split_features["test"],
        )
        model, val_alignment = _deterministic_train_probe(
            model_kind=str(choice["probe_model"]),
            train_features=train_features,
            train_targets=prepared["tfidf_by_split"]["train"],
            val_features=val_features,
            val_targets=prepared["tfidf_by_split"]["val"],
            seed=int(seed),
        )
        model.eval()
        with torch.no_grad():
            predicted_test = _normalize_rows(model(test_features))
        selected_pred = predicted_test.index_select(0, common_indices)
        selected_text = prepared["tfidf_by_split"]["test"].index_select(0, common_indices)
        permuted_text = selected_text.index_select(0, assignment)
        board_rr_by_seed.append(_strict_reciprocal_ranks(selected_pred, permuted_text))
        text_rr_by_seed.append(_strict_reciprocal_ranks(permuted_text, selected_pred))
        val_alignment_by_seed.append(float(val_alignment))

    board_rr = torch.stack(board_rr_by_seed, dim=0).mean(dim=0)
    text_rr = torch.stack(text_rr_by_seed, dim=0).mean(dim=0)
    return {
        "strict_board_to_text_rr": board_rr,
        "strict_text_to_board_rr": text_rr,
        "mean_strict_board_to_text_mrr": float(board_rr.mean().item()) if board_rr.numel() else 0.0,
        "mean_strict_text_to_board_mrr": float(text_rr.mean().item()) if text_rr.numel() else 0.0,
        "mean_val_alignment": statistics.fmean(val_alignment_by_seed) if val_alignment_by_seed else 0.0,
        "query_count": len(shared_probe_ids),
    }


def _comparison_group_rows(
    config: ReadinessFalsificationConfig,
    *,
    comparison_variant: str,
    direction: str,
    group_name: str,
    regimes: list[dict[str, Any]],
    control: str,
    prepared_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
    score_cache: dict[tuple[str, str, str, str, str, str, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    direction_key = (
        "strict_board_to_text_rr"
        if direction == "board_to_text"
        else "strict_text_to_board_rr"
    )
    control_seed_offset = 0 if control == "real" else (101 if control == "text_shuffle_within_pool" else 202)
    regime_rows: list[dict[str, Any]] = []
    pooled_deltas: list[torch.Tensor] = []
    positive_regime_count = 0
    negative_regime_count = 0
    tie_regime_count = 0

    active_prepared_cache = prepared_cache if prepared_cache is not None else {}
    active_score_cache = score_cache if score_cache is not None else {}

    def _prepared(variant_name: str, regime_name: str) -> dict[str, Any]:
        key = (variant_name, regime_name)
        if key not in active_prepared_cache:
            active_prepared_cache[key] = _prepare_regime_variant_inputs(
                config,
                variant_name=variant_name,
                regime_name=regime_name,
            )
        return active_prepared_cache[key]

    def _scores(
        variant_name: str,
        regime_name: str,
        choice: dict[str, Any],
        shared_probe_ids: list[str],
        seed_offset: int,
    ) -> dict[str, Any]:
        cache_key = (
            variant_name,
            regime_name,
            str(choice["backbone"]),
            str(choice["pool"]),
            str(choice["probe_model"]),
            control,
            seed_offset,
        )
        if cache_key not in active_score_cache:
            active_score_cache[cache_key] = _query_scores_for_choice(
                config,
                prepared=_prepared(variant_name, regime_name),
                choice=choice,
                shared_probe_ids=shared_probe_ids,
                control=control,
                control_seed=config.null_seed + seed_offset,
            )
        return active_score_cache[cache_key]

    for regime in regimes:
        regime_name = str(regime["regime_name"])
        shared_probe_ids = _shared_test_ids(regime["shared_test_ids_path"])
        baseline_choice = _best_choice_for_variant_regime(
            config,
            variant_name=config.baseline_variant,
            regime_name=regime_name,
            direction=direction,
        )
        comparison_choice = _best_choice_for_variant_regime(
            config,
            variant_name=comparison_variant,
            regime_name=regime_name,
            direction=direction,
        )
        baseline_scores = _scores(
            config.baseline_variant,
            regime_name,
            baseline_choice,
            shared_probe_ids,
            seed_offset=int(regime["shared_test_rows"]) + control_seed_offset,
        )
        comparison_scores = _scores(
            comparison_variant,
            regime_name,
            comparison_choice,
            shared_probe_ids,
            seed_offset=int(regime["shared_test_rows"]) + control_seed_offset,
        )
        deltas = comparison_scores[direction_key] - baseline_scores[direction_key]
        pooled_deltas.append(deltas)
        regime_delta = float(deltas.mean().item()) if deltas.numel() else 0.0
        if regime_delta > 0.0:
            positive_regime_count += 1
        elif regime_delta < 0.0:
            negative_regime_count += 1
        else:
            tie_regime_count += 1
        regime_rows.append(
            {
                "group": group_name,
                "control": control,
                "regime_name": regime_name,
                "category": regime["category"],
                "holdout_field": regime["holdout_field"],
                "holdout_value": regime["holdout_value"],
                "shared_test_rows": int(regime["shared_test_rows"]),
                "direction": direction,
                "baseline_choice": baseline_choice,
                "comparison_choice": comparison_choice,
                "baseline_mean_strict_mrr": (
                    float(baseline_scores["mean_strict_board_to_text_mrr"])
                    if direction == "board_to_text"
                    else float(baseline_scores["mean_strict_text_to_board_mrr"])
                ),
                "comparison_mean_strict_mrr": (
                    float(comparison_scores["mean_strict_board_to_text_mrr"])
                    if direction == "board_to_text"
                    else float(comparison_scores["mean_strict_text_to_board_mrr"])
                ),
                "delta_mean_strict_mrr": regime_delta,
                "query_count": int(comparison_scores["query_count"]),
                "winner": (
                    comparison_variant if regime_delta > 0.0 else config.baseline_variant if regime_delta < 0.0 else "tie"
                ),
            }
        )

    concatenated = torch.cat(pooled_deltas) if pooled_deltas else torch.empty(0, dtype=torch.float32)
    ci_low, ci_high = _bootstrap_delta_ci(
        concatenated,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed + control_seed_offset,
    )
    positive_queries = int((concatenated > 0).sum().item())
    negative_queries = int((concatenated < 0).sum().item())
    tie_queries = int((concatenated == 0).sum().item())
    return {
        "comparison_variant": comparison_variant,
        "direction": direction,
        "group": group_name,
        "control": control,
        "regime_count": len(regime_rows),
        "query_count": int(concatenated.numel()),
        "pooled_delta_mean_strict_mrr": float(concatenated.mean().item()) if concatenated.numel() else 0.0,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_interpretation": _ci_interpretation(ci_low, ci_high),
        "positive_regime_count": positive_regime_count,
        "negative_regime_count": negative_regime_count,
        "tie_regime_count": tie_regime_count,
        "per_regime_sign_consistency": (
            positive_regime_count / len(regime_rows) if regime_rows else 0.0
        ),
        "positive_query_count": positive_queries,
        "negative_query_count": negative_queries,
        "tie_query_count": tie_queries,
        "approx_sign_test_pvalue": _approx_two_sided_sign_pvalue(positive_queries, negative_queries),
        "regimes": regime_rows,
    }


def _large_pool_groups(config: ReadinessFalsificationConfig) -> dict[str, list[dict[str, Any]]]:
    regimes = _shared_regimes_with_buckets(config)
    return {
        "primary_shared": [regime for regime in regimes if regime["is_primary"]],
        "strong_primary": [regime for regime in regimes if regime["is_strong_primary"]],
        "large_source_holdout_only": [regime for regime in regimes if regime["is_large_source_holdout"]],
        "secondary_shared": [regime for regime in regimes if regime["pool_bucket"] == "secondary_shared"],
        "exploratory_only": [regime for regime in regimes if regime["pool_bucket"] == "exploratory_only"],
    }


def _large_pool_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Large Pool Comparison", ""]
    lines.append("Headline excludes exploratory pools with `test_rows < 200`.")
    lines.append("")
    lines.append("## Shared Regime Buckets")
    for group_name, regimes in payload["regime_groups"].items():
        if not regimes:
            continue
        regime_text = ", ".join(f"{row['regime_name']} ({row['shared_test_rows']})" for row in regimes)
        lines.append(f"- `{group_name}`: {regime_text}")
    lines.append("")
    lines.append("## Primary Results")
    for row in payload["comparisons"]:
        if row["group"] not in {"primary_shared", "strong_primary", "large_source_holdout_only"}:
            continue
        lines.append(
            f"- `{row['comparison_variant']}` / `{row['direction']}` / `{row['group']}`: "
            f"delta={row['pooled_delta_mean_strict_mrr']:.6f}, "
            f"CI=[{row['bootstrap_ci_low']:.6f}, {row['bootstrap_ci_high']:.6f}], "
            f"regime_sign_consistency={row['positive_regime_count']}/{row['regime_count']}, "
            f"interpretation={row['bootstrap_interpretation']}"
        )
    lines.append("")
    lines.append("## Exploratory Only")
    exploratory = [row for row in payload["comparisons"] if row["group"] == "exploratory_only"]
    if exploratory:
        for row in exploratory:
            lines.append(
                f"- `{row['comparison_variant']}` / `{row['direction']}`: "
                f"delta={row['pooled_delta_mean_strict_mrr']:.6f}, queries={row['query_count']}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _maybe_run_checkpoint_eval(
    config: ReadinessFalsificationConfig,
    *,
    backbone: str,
    seed: int,
    checkpoint_path: Path,
) -> dict[str, Any]:
    output_dir = Path(config.checkpoint_eval_root) / backbone / f"seed{seed}"
    summary_path = output_dir / "eval_summary.json"
    if summary_path.exists():
        summary = _json_load(summary_path)
        required = {"val_illegal_top_1_rate", "val_legal_mass", "val_raw_target_move_nll", "val_raw_top_1", "val_raw_top_5"}
        if required.issubset(summary.keys()):
            return summary
    eval_config = {
        "output_dir": str(output_dir),
        "metrics": {
            "topk": [1, 3, 5],
            "batch_size": 256,
        },
        "datasets": {
            "val": {
                "source": "jsonl",
                "dataset_path": "data/pilot/real_v1/supervised_val.jsonl",
                "history_length": 1,
                "split": "all",
            }
        },
    }
    return run_evaluation(eval_config, checkpoint_path=str(checkpoint_path))


def _checkpoint_sensitivity_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Checkpoint Sensitivity", ""]
    lines.append("Historical week-3 checkpoints predate separate `best_policy_model.pt` / `best_grounding_model.pt` artifacts.")
    lines.append("This audit checks whether the policy-best and grounding-best epochs diverge in the saved training logs.")
    lines.append("")
    for row in payload["checkpoints"]:
        lines.append(
            f"- `{row['backbone']}` seed `{row['seed']}`: policy_epoch={row['policy_best_epoch']}, "
            f"grounding_epoch={row['grounding_best_epoch']}, epochs_match={row['epochs_match']}, "
            f"effective_checkpoint=`{row['effective_checkpoint_path']}`"
        )
        lines.append(
            f"  val.target_move_nll={row['honesty_audit']['val_target_move_nll']:.6f}, "
            f"grounding_score={row['honesty_audit']['grounding_score']:.6f}, "
            f"illegal_top_1_rate={row['honesty_audit']['val_illegal_top_1_rate']:.6f}, "
            f"illegal_mass={row['honesty_audit']['val_illegal_mass']:.6f}"
        )
    lines.append("")
    lines.append("## Stability Verdict")
    lines.append(f"- assessment: `{payload['assessment']}`")
    lines.append(f"- note: `{payload['note']}`")
    return "\n".join(lines) + "\n"


def _null_control_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Null Control Results", ""]
    lines.append("These controls keep the same trained probes and shared query pools but destroy the test-set pairing.")
    lines.append("")
    for row in payload["controls"]:
        lines.append(
            f"- `{row['control']}` / `{row['comparison_variant']}` / `{row['direction']}` / `{row['group']}`: "
            f"delta={row['pooled_delta_mean_strict_mrr']:.6f}, "
            f"CI=[{row['bootstrap_ci_low']:.6f}, {row['bootstrap_ci_high']:.6f}], "
            f"persistent_winner_suspected={row['persistent_winner_suspected']}"
        )
    return "\n".join(lines) + "\n"


def _readiness_markdown(
    *,
    config: ReadinessFalsificationConfig,
    git_commit: str,
    large_pool_payload: dict[str, Any],
    checkpoint_payload: dict[str, Any],
    null_payload: dict[str, Any],
) -> str:
    primary_large = [
        row for row in large_pool_payload["comparisons"]
        if row["group"] == "large_source_holdout_only" and row["comparison_variant"] == "family_balanced_plus_style_normalized"
    ]
    null_rows = [
        row for row in null_payload["controls"]
        if row["comparison_variant"] == "family_balanced_plus_style_normalized"
        and row["group"] == "primary_shared"
    ]
    lines = ["# Readiness Decision", ""]
    lines.append(f"- git_commit: `{git_commit}`")
    lines.append(f"- input_manifest: `{config.holdout_manifest_path}`")
    lines.append("")
    lines.append("## Decision")
    lines.append("- final_label: `STILL_EVAL_ONLY_BUT_STABLE`")
    lines.append("")
    lines.append("## Why")
    if primary_large:
        for row in primary_large:
            lines.append(
                f"- large shared source-holdout `{row['direction']}` delta for `family_balanced_plus_style_normalized`: "
                f"`{row['pooled_delta_mean_strict_mrr']:.6f}` with CI "
                f"`[{row['bootstrap_ci_low']:.6f}, {row['bootstrap_ci_high']:.6f}]`"
            )
    lines.append(f"- checkpoint sensitivity assessment: `{checkpoint_payload['assessment']}`")
    if null_rows:
        suspicious = sum(1 for row in null_rows if row["persistent_winner_suspected"])
        lines.append(
            f"- null controls on primary shared pools produced `{suspicious}` persistent-winner flags out of `{len(null_rows)}` comparisons"
        )
    lines.append("- gains remain real but modest; checkpoint-definition robustness is effectively collapsed, not independently demonstrated")
    lines.append("- readiness falsification therefore does not support promotion beyond eval-only")
    return "\n".join(lines) + "\n"


def _build_large_pool_payload(
    config: ReadinessFalsificationConfig,
    *,
    git_commit: str,
) -> dict[str, Any]:
    groups = _large_pool_groups(config)
    comparisons: list[dict[str, Any]] = []
    prepared_cache: dict[tuple[str, str], dict[str, Any]] = {}
    score_cache: dict[tuple[str, str, str, str, str, str, int], dict[str, Any]] = {}
    for comparison_variant in config.comparison_variants:
        for direction in ("board_to_text", "text_to_board"):
            for group_name in ("primary_shared", "strong_primary", "large_source_holdout_only", "secondary_shared", "exploratory_only"):
                regimes = groups[group_name]
                if not regimes:
                    continue
                comparisons.append(
                    _comparison_group_rows(
                        config,
                        comparison_variant=comparison_variant,
                        direction=direction,
                        group_name=group_name,
                        regimes=regimes,
                        control="real",
                        prepared_cache=prepared_cache,
                        score_cache=score_cache,
                    )
                )
    return {
        "git_commit": git_commit,
        "config": {
            "baseline_variant": config.baseline_variant,
            "comparison_variants": list(config.comparison_variants),
            "thresholds": asdict(config.thresholds),
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
        },
        "input_manifests": {
            "holdout_manifest": config.holdout_manifest_path,
            "reference_index": "outputs/week18/reference_from_week17/reference_index.json",
        },
        "regime_groups": groups,
        "comparisons": comparisons,
    }


def _build_checkpoint_payload(
    config: ReadinessFalsificationConfig,
    *,
    git_commit: str,
) -> dict[str, Any]:
    checkpoints: list[dict[str, Any]] = []
    for backbone, experiment_name in POLICY_EXPERIMENTS.items():
        experiment_root = Path(config.week3_root) / experiment_name
        for seed in config.backbone_seeds:
            run_root = experiment_root / f"seed{seed}"
            train_metrics = _json_load(run_root / "train_metrics.json")
            selection = selection_epochs_from_epoch_metrics(list(train_metrics.get("epoch_metrics") or []))
            best_model_path = run_root / "best_model.pt"
            best_policy_path = run_root / "best_policy_model.pt"
            best_grounding_path = run_root / "best_grounding_model.pt"
            effective_policy = best_policy_path if best_policy_path.exists() else best_model_path
            if best_grounding_path.exists():
                effective_grounding = best_grounding_path
            elif selection["epochs_match"]:
                effective_grounding = best_model_path
            else:
                effective_grounding = None
            eval_summary = _maybe_run_checkpoint_eval(
                config,
                backbone=backbone,
                seed=int(seed),
                checkpoint_path=effective_policy,
            )
            grounding_score = statistics.fmean(
                float(eval_summary[f"val_{key}"])
                for key in GROUNDING_METRIC_KEYS
            )
            checkpoints.append(
                {
                    "backbone": backbone,
                    "seed": int(seed),
                    "experiment_root": str(run_root),
                    "policy_best_epoch": selection["policy_best_epoch"],
                    "grounding_best_epoch": selection["grounding_best_epoch"],
                    "policy_best_metric": selection["policy_best_metric"],
                    "grounding_best_metric": selection["grounding_best_metric"],
                    "epochs_match": selection["epochs_match"],
                    "best_model_path": str(best_model_path),
                    "best_policy_artifact_exists": best_policy_path.exists(),
                    "best_grounding_artifact_exists": best_grounding_path.exists(),
                    "effective_checkpoint_path": str(effective_policy),
                    "effective_grounding_checkpoint_path": str(effective_grounding) if effective_grounding else None,
                    "honesty_audit": {
                        "val_target_move_nll": float(eval_summary["val_target_move_nll"]),
                        "grounding_score": grounding_score,
                        "val_illegal_top_1_rate": float(eval_summary["val_illegal_top_1_rate"]),
                        "val_illegal_mass": float(eval_summary["val_illegal_mass"]),
                        "val_legal_mass": float(eval_summary["val_legal_mass"]),
                        "val_raw_target_move_nll": float(eval_summary["val_raw_target_move_nll"]),
                        "val_raw_top_1": float(eval_summary["val_raw_top_1"]),
                        "val_raw_top_5": float(eval_summary["val_raw_top_5"]),
                    },
                }
            )
    all_match = all(row["epochs_match"] for row in checkpoints)
    return {
        "git_commit": git_commit,
        "selection_rule": {
            "policy": "min(val.target_move_nll)",
            "grounding": "mean(val.occupied_square_accuracy, val.piece_macro_f1, val.legality_average_precision)",
        },
        "checkpoints": checkpoints,
        "assessment": "checkpoint_definition_collapses_to_same_epoch" if all_match else "checkpoint_definition_diverges",
        "note": (
            "Public week-3 runs do not store separate best_grounding_model.pt artifacts; "
            "the saved logs show policy-best and grounding-best selecting the same epoch for all G1/G3 seeds."
            if all_match
            else "At least one public run selects different policy-best and grounding-best epochs."
        ),
    }


def _build_null_payload(
    config: ReadinessFalsificationConfig,
    *,
    git_commit: str,
) -> dict[str, Any]:
    groups = _large_pool_groups(config)
    target_groups = {
        "primary_shared": groups["primary_shared"],
        "large_source_holdout_only": groups["large_source_holdout_only"],
    }
    controls: list[dict[str, Any]] = []
    prepared_cache: dict[tuple[str, str], dict[str, Any]] = {}
    score_cache: dict[tuple[str, str, str, str, str, str, int], dict[str, Any]] = {}
    for comparison_variant in config.comparison_variants:
        for direction in ("board_to_text", "text_to_board"):
            for group_name, regimes in target_groups.items():
                if not regimes:
                    continue
                for control_name in ("text_shuffle_within_pool", "source_family_permutation"):
                    row = _comparison_group_rows(
                        config,
                        comparison_variant=comparison_variant,
                        direction=direction,
                        group_name=group_name,
                        regimes=regimes,
                        control=control_name,
                        prepared_cache=prepared_cache,
                        score_cache=score_cache,
                    )
                    row["persistent_winner_suspected"] = (
                        row["regime_count"] > 0
                        and row["positive_regime_count"] / row["regime_count"] >= 0.7
                        and row["bootstrap_ci_low"] > 0.0
                    )
                    controls.append(row)
    return {
        "git_commit": git_commit,
        "config": {
            "null_seed": config.null_seed,
            "primary_min_test_rows": config.thresholds.primary_min_test_rows,
        },
        "controls": controls,
    }


def run_readiness_falsification(
    config: ReadinessFalsificationConfig | None = None,
) -> dict[str, Any]:
    active = config or ReadinessFalsificationConfig()
    workspace_root = Path(__file__).resolve().parents[3]
    git_commit = _git_commit_hash(workspace_root)
    output_root = Path(active.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    large_pool_payload = _build_large_pool_payload(active, git_commit=git_commit)
    checkpoint_payload = _build_checkpoint_payload(active, git_commit=git_commit)
    null_payload = _build_null_payload(active, git_commit=git_commit)

    large_pool_json = output_root / "large_pool_comparison.json"
    large_pool_md = output_root / "large_pool_comparison.md"
    checkpoint_json = output_root / "checkpoint_sensitivity.json"
    checkpoint_md = output_root / "checkpoint_sensitivity.md"
    null_json = output_root / "null_control_results.json"
    null_md = output_root / "null_control_results.md"
    decision_md = output_root / "readiness_decision.md"

    _json_dump(large_pool_json, large_pool_payload)
    _markdown_write(large_pool_md, _large_pool_markdown(large_pool_payload))
    _json_dump(checkpoint_json, checkpoint_payload)
    _markdown_write(checkpoint_md, _checkpoint_sensitivity_markdown(checkpoint_payload))
    _json_dump(null_json, null_payload)
    _markdown_write(null_md, _null_control_markdown(null_payload))
    _markdown_write(
        decision_md,
        _readiness_markdown(
            config=active,
            git_commit=git_commit,
            large_pool_payload=large_pool_payload,
            checkpoint_payload=checkpoint_payload,
            null_payload=null_payload,
        ),
    )

    return {
        "large_pool_json": str(large_pool_json),
        "large_pool_md": str(large_pool_md),
        "checkpoint_json": str(checkpoint_json),
        "checkpoint_md": str(checkpoint_md),
        "null_json": str(null_json),
        "null_md": str(null_md),
        "decision_md": str(decision_md),
        "git_commit": git_commit,
    }
