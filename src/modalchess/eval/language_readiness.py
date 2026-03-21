"""Week-5 frozen-embedding language readiness probes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from modalchess.utils.device import resolve_device


def _load_target_rows(path: str | Path) -> list[dict[str, Any]]:
    target_path = Path(path)
    rows: list[dict[str, Any]] = []
    with target_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_embedding_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"embedding payload가 dict가 아니다: {path}")
    return payload


def _align_embeddings_and_targets(
    embedding_payload: dict[str, Any],
    target_rows: list[dict[str, Any]],
) -> tuple[list[str], torch.Tensor, torch.Tensor, list[str]]:
    target_by_probe_id = {str(row["probe_id"]): row for row in target_rows}
    probe_ids = [str(probe_id) for probe_id in embedding_payload["probe_id"]]
    available_probe_ids = [probe_id for probe_id in probe_ids if probe_id in target_by_probe_id]
    if not available_probe_ids:
        return [], torch.empty(0), torch.empty(0), []
    probe_id_to_index = {str(probe_id): index for index, probe_id in enumerate(probe_ids)}
    first_rows = [target_by_probe_id[probe_id] for probe_id in available_probe_ids]
    label_vocab = sorted({label for row in first_rows for label in row.get("target_labels", [])})
    index_tensor = torch.tensor([probe_id_to_index[probe_id] for probe_id in available_probe_ids], dtype=torch.long)
    return available_probe_ids, index_tensor, _target_matrix(first_rows, label_vocab), label_vocab


def _target_matrix(rows: list[dict[str, Any]], label_vocab: list[str]) -> torch.Tensor:
    label_to_index = {label: index for index, label in enumerate(label_vocab)}
    matrix = torch.zeros((len(rows), len(label_vocab)), dtype=torch.float32)
    for row_index, row in enumerate(rows):
        for label in row.get("target_labels", []):
            if label in label_to_index:
                matrix[row_index, label_to_index[label]] = 1.0
    return matrix


def _filter_label_vocab(
    train_rows: list[dict[str, Any]],
    min_train_positive: int,
) -> list[str]:
    counts: dict[str, int] = {}
    for row in train_rows:
        for label in row.get("target_labels", []):
            counts[str(label)] = counts.get(str(label), 0) + 1
    return sorted([label for label, count in counts.items() if count >= min_train_positive])


def _matrix_for_rows(rows: list[dict[str, Any]], label_vocab: list[str]) -> torch.Tensor:
    return _target_matrix(rows, label_vocab)


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


def _average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
    positives = int(targets.sum().item())
    if positives == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    ranked_targets = targets[order]
    true_positive_cumsum = torch.cumsum(ranked_targets, dim=0)
    ranks = torch.arange(1, ranked_targets.numel() + 1, dtype=torch.float32)
    precision = true_positive_cumsum / ranks
    return float((precision * ranked_targets).sum().item() / positives)


def _multilabel_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    if targets.numel() == 0 or targets.size(1) == 0:
        return {
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "micro_average_precision": 0.0,
            "macro_average_precision": 0.0,
        }
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()
    true_positive = (predictions * targets).sum().item()
    false_positive = (predictions * (1.0 - targets)).sum().item()
    false_negative = ((1.0 - predictions) * targets).sum().item()
    micro_precision = true_positive / (true_positive + false_positive + 1e-8)
    micro_recall = true_positive / (true_positive + false_negative + 1e-8)
    micro_f1 = 2.0 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)

    per_label_f1: list[float] = []
    per_label_ap: list[float] = []
    for column_index in range(targets.size(1)):
        label_targets = targets[:, column_index]
        label_predictions = predictions[:, column_index]
        label_tp = (label_predictions * label_targets).sum().item()
        label_fp = (label_predictions * (1.0 - label_targets)).sum().item()
        label_fn = ((1.0 - label_predictions) * label_targets).sum().item()
        label_precision = label_tp / (label_tp + label_fp + 1e-8)
        label_recall = label_tp / (label_tp + label_fn + 1e-8)
        label_f1 = 2.0 * label_precision * label_recall / (label_precision + label_recall + 1e-8)
        per_label_f1.append(label_f1)
        per_label_ap.append(_average_precision(probabilities[:, column_index], label_targets))

    micro_ap = _average_precision(probabilities.reshape(-1), targets.reshape(-1))
    macro_ap = statistics.fmean(per_label_ap) if per_label_ap else 0.0
    macro_f1 = statistics.fmean(per_label_f1) if per_label_f1 else 0.0
    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "micro_average_precision": float(micro_ap),
        "macro_average_precision": float(macro_ap),
    }


def _constant_baseline_scores(train_targets: torch.Tensor, num_rows: int) -> torch.Tensor:
    prevalence = train_targets.mean(dim=0, keepdim=True)
    prevalence = prevalence.clamp(min=1e-4, max=1.0 - 1e-4)
    return torch.logit(prevalence).repeat(num_rows, 1)


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(inputs)


class MlpProbe(nn.Module):
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
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(features.size(0), generator=generator)[:limit]
    return features[indices], targets[indices]


def _train_probe(
    *,
    model_kind: str,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    seed: int,
    max_train_rows: int | None = None,
) -> tuple[nn.Module, dict[str, float]]:
    device = resolve_device()
    if max_train_rows is not None:
        train_features, train_targets = _subsample_tensors(train_features, train_targets, max_train_rows, seed)
    input_dim = train_features.size(1)
    output_dim = train_targets.size(1)
    if model_kind == "linear":
        model: nn.Module = LinearProbe(input_dim=input_dim, output_dim=output_dim)
        learning_rate = 5e-3
        max_epochs = 12
    elif model_kind == "mlp":
        model = MlpProbe(input_dim=input_dim, output_dim=output_dim)
        learning_rate = 1e-3
        max_epochs = 20
    else:
        raise ValueError(f"지원하지 않는 probe model: {model_kind}")

    model = model.to(device)
    pos_counts = train_targets.sum(dim=0)
    neg_counts = train_targets.size(0) - pos_counts
    pos_weight = (neg_counts / pos_counts.clamp(min=1.0)).clamp(max=32.0).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=1024,
        shuffle=True,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_metric = float("-inf")
    best_metrics: dict[str, float] = {}
    patience = 3
    patience_left = patience

    for _epoch in range(max_epochs):
        model.train()
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_features.to(device)).cpu()
        val_metrics = _multilabel_metrics(val_logits, val_targets)
        metric_value = val_metrics["micro_average_precision"]
        if metric_value > best_val_metric:
            best_val_metric = metric_value
            best_metrics = val_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), best_metrics


def _evaluate_probe(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(features)
    return _multilabel_metrics(logits, targets)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _result_metrics() -> tuple[str, ...]:
    return (
        "val_micro_f1",
        "val_macro_f1",
        "val_micro_average_precision",
        "val_macro_average_precision",
        "test_micro_f1",
        "test_macro_f1",
        "test_micro_average_precision",
        "test_macro_average_precision",
    )


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["family"]),
            str(row["backbone"]),
            str(row["pool"]),
            str(row["probe_model"]),
        )
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
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
            "train_rows_mean": statistics.fmean(float(row["train_rows"]) for row in group_rows),
            "val_rows_mean": statistics.fmean(float(row["val_rows"]) for row in group_rows),
            "test_rows_mean": statistics.fmean(float(row["test_rows"]) for row in group_rows),
        }
        for metric_name in _result_metrics():
            metric_values = [float(row[metric_name]) for row in group_rows]
            aggregate_row[f"{metric_name}_mean"] = statistics.fmean(metric_values)
            aggregate_row[f"{metric_name}_std"] = statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _best_row_for_family(
    rows: list[dict[str, Any]],
    family: str,
    *,
    baseline: bool,
) -> dict[str, Any] | None:
    family_rows = [
        row
        for row in rows
        if row["family"] == family and ((row["probe_model"] == "baseline") == baseline)
    ]
    if not family_rows:
        return None
    return max(
        family_rows,
        key=lambda row: (
            float(row.get("test_micro_average_precision_mean", row.get("test_micro_average_precision", 0.0))),
            float(row.get("test_micro_f1_mean", row.get("test_micro_f1", 0.0))),
        ),
    )


def _family_signal_summary(aggregate_rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    best_probe = _best_row_for_family(aggregate_rows, family, baseline=False)
    best_baseline = _best_row_for_family(aggregate_rows, family, baseline=True)
    if best_probe is None or best_baseline is None:
        return {
            "family": family,
            "has_signal": False,
            "ap_gain": 0.0,
            "f1_gain": 0.0,
        }
    ap_gain = float(best_probe["test_micro_average_precision_mean"] - best_baseline["test_micro_average_precision_mean"])
    f1_gain = float(best_probe["test_micro_f1_mean"] - best_baseline["test_micro_f1_mean"])
    return {
        "family": family,
        "has_signal": ap_gain > 0.02 or f1_gain > 0.05,
        "strong_signal": ap_gain > 0.06 and f1_gain > 0.10,
        "ap_gain": ap_gain,
        "f1_gain": f1_gain,
        "best_probe": best_probe,
        "best_baseline": best_baseline,
    }


def _decide_week7_state(aggregate_rows: list[dict[str, Any]]) -> str:
    puzzle_signal = _family_signal_summary(aggregate_rows, "puzzle")
    mate_signal = _family_signal_summary(aggregate_rows, "mate")
    if puzzle_signal.get("strong_signal") and mate_signal.get("strong_signal"):
        return "READY_FOR_TINY_FROZEN_ALIGNMENT"
    if puzzle_signal.get("has_signal") and mate_signal.get("has_signal"):
        return "STILL_EVAL_ONLY_BUT_STABLE"
    return "DATA_EXPANSION_FIRST"


def _pool_winner(aggregate_rows: list[dict[str, Any]], family: str) -> str | None:
    candidate_rows = [row for row in aggregate_rows if row["family"] == family and row["probe_model"] != "baseline"]
    if not candidate_rows:
        return None
    best_row = max(candidate_rows, key=lambda row: row["test_micro_average_precision_mean"])
    return str(best_row["pool"])


def _summary_markdown(state_name: str, rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> str:
    lines = ["# Readiness Probe Summary", ""]
    lines.append(f"- week7_state_candidate: `{state_name}`")
    lines.append("")
    lines.append("## Key Findings")
    for family in ("puzzle", "mate"):
        family_signal = _family_signal_summary(aggregate_rows, family)
        best_row = family_signal.get("best_probe")
        best_baseline = family_signal.get("best_baseline")
        if best_row is None or best_baseline is None:
            continue
        lines.append(
            f"- `{family}` best probe: backbone={best_row['backbone']}, pool={best_row['pool']}, "
            f"probe={best_row['probe_model']}, test_micro_ap={best_row['test_micro_average_precision_mean']:.4f} +/- {best_row['test_micro_average_precision_std']:.4f}, "
            f"test_micro_f1={best_row['test_micro_f1_mean']:.4f} +/- {best_row['test_micro_f1_std']:.4f}"
        )
        lines.append(
            f"- `{family}` baseline comparison: ap_gain={family_signal['ap_gain']:.4f}, "
            f"f1_gain={family_signal['f1_gain']:.4f}"
        )
    lines.append("")
    lines.append("## Summary Answers")
    puzzle_signal = _family_signal_summary(aggregate_rows, "puzzle")
    mate_signal = _family_signal_summary(aggregate_rows, "mate")
    lines.append(
        "- Puzzle signal above baseline after v2 split hygiene and multi-seed evaluation: "
        f"{'yes' if puzzle_signal.get('has_signal') else 'no'}"
    )
    lines.append(
        "- MATE signal above baseline after v2 split hygiene and multi-seed evaluation: "
        f"{'yes' if mate_signal.get('has_signal') else 'no'}"
    )
    puzzle_best = puzzle_signal.get("best_probe")
    mate_best = mate_signal.get("best_probe")
    lines.append(
        "- G1 vs G3 pattern: "
        f"puzzle_best={puzzle_best['backbone'] if puzzle_best else 'n/a'}, "
        f"mate_best={mate_best['backbone'] if mate_best else 'n/a'}"
    )
    lines.append(
        "- Board vs context pooled conclusion: "
        f"puzzle_best_pool={_pool_winner(aggregate_rows, 'puzzle') or 'n/a'}, "
        f"mate_best_pool={_pool_winner(aggregate_rows, 'mate') or 'n/a'}"
    )
    lines.append(
        "- Results strong enough to justify future light alignment work: "
        f"{'yes, but still eval-only by default' if state_name != 'DATA_EXPANSION_FIRST' else 'no'}"
    )
    return "\n".join(lines) + "\n"


def run_language_readiness_probes(
    *,
    embedding_root: str | Path,
    target_root: str | Path,
    output_dir: str | Path,
    backbone_seed: int = 11,
    backbone_seeds: list[int] | None = None,
    mate_min_train_positive: int = 25,
    puzzle_min_train_positive: int = 25,
) -> dict[str, Any]:
    """Run frozen language-readiness probes with optional multi-seed aggregation."""
    embedding_root_path = Path(embedding_root)
    target_root_path = Path(target_root)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    seed_list = backbone_seeds or [backbone_seed]
    results: list[dict[str, Any]] = []
    for family, min_train_positive in (("mate", mate_min_train_positive), ("puzzle", puzzle_min_train_positive)):
        target_rows_by_split = {
            split_name: _load_target_rows(target_root_path / f"{family}_targets_{split_name}.jsonl")
            for split_name in ("train", "val", "test")
        }
        label_vocab = _filter_label_vocab(target_rows_by_split["train"], min_train_positive=min_train_positive)
        if not label_vocab:
            continue
        target_matrices = {
            split_name: _matrix_for_rows(target_rows_by_split[split_name], label_vocab)
            for split_name in ("train", "val", "test")
        }
        probe_ids_by_split = {
            split_name: [str(row["probe_id"]) for row in target_rows_by_split[split_name]]
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
                    baseline_val_logits = _constant_baseline_scores(target_matrices["train"], target_matrices["val"].size(0))
                    baseline_test_logits = _constant_baseline_scores(target_matrices["train"], target_matrices["test"].size(0))
                    results.append(
                        {
                            "family": family,
                            "backbone": backbone_name,
                            "seed": seed,
                            "pool": pool_name,
                            "probe_model": "baseline",
                            "label_count": len(label_vocab),
                            "train_rows": int(train_features.size(0)),
                            "val_rows": int(val_features.size(0)),
                            "test_rows": int(test_features.size(0)),
                            **{f"val_{key}": value for key, value in _multilabel_metrics(baseline_val_logits, target_matrices["val"]).items()},
                            **{f"test_{key}": value for key, value in _multilabel_metrics(baseline_test_logits, target_matrices["test"]).items()},
                        }
                    )

                    for probe_model, train_limit in (("linear", None), ("mlp", 100000 if family == "mate" else 50000)):
                        trained_model, val_metrics = _train_probe(
                            model_kind=probe_model,
                            train_features=train_features,
                            train_targets=target_matrices["train"],
                            val_features=val_features,
                            val_targets=target_matrices["val"],
                            seed=seed,
                            max_train_rows=train_limit,
                        )
                        test_metrics = _evaluate_probe(trained_model, test_features, target_matrices["test"])
                        results.append(
                            {
                                "family": family,
                                "backbone": backbone_name,
                                "seed": seed,
                                "pool": pool_name,
                                "probe_model": probe_model,
                                "label_count": len(label_vocab),
                                "train_rows": int(train_features.size(0)),
                                "val_rows": int(val_features.size(0)),
                                "test_rows": int(test_features.size(0)),
                                **{f"val_{key}": value for key, value in val_metrics.items()},
                                **{f"test_{key}": value for key, value in test_metrics.items()},
                            }
                        )

    aggregate_rows = _aggregate_rows(results)
    state_name = _decide_week7_state(aggregate_rows)
    output_payload = {
        "week7_state_candidate": state_name,
        "results": results,
        "aggregate": aggregate_rows,
    }
    json_path = output_root / "probe_results.json"
    csv_path = output_root / "probe_results.csv"
    aggregate_csv_path = output_root / "probe_results_aggregate.csv"
    summary_path = output_root / "readiness_probe_summary.md"
    json_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    _write_csv(csv_path, results)
    _write_csv(aggregate_csv_path, aggregate_rows)
    summary_path.write_text(_summary_markdown(state_name, results, aggregate_rows), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "aggregate_csv_path": str(aggregate_csv_path),
        "summary_path": str(summary_path),
        "week7_state_candidate": state_name,
        "results": results,
        "aggregate": aggregate_rows,
    }
