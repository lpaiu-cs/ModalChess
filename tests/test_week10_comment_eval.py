from __future__ import annotations

import json
from pathlib import Path

import yaml

from modalchess.data.comment_retrieval_eval import (
    CommentRetrievalEvalConfig,
    build_comment_retrieval_eval_regime,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_probe_ids(path: Path) -> list[str]:
    probe_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                probe_ids.append(str(json.loads(line)["probe_id"]))
    return probe_ids


def test_comment_retrieval_eval_regime_is_deterministic_and_documented(tmp_path: Path) -> None:
    input_root = tmp_path / "annotated_sidecar_v1"
    rows = {
        "train": [
            {"probe_id": f"train_{index}", "comment_text": f"comment {index}", "comment_source": "comment"}
            for index in range(6)
        ]
        + [
            {"probe_id": f"train_nag_{index}", "comment_text": f"nag comment {index}", "comment_source": "comment+nag"}
            for index in range(2)
        ]
        + [{"probe_id": "train_empty", "comment_text": "", "comment_source": "nag"}],
        "val": [
            {"probe_id": f"val_{index}", "comment_text": f"comment {index}", "comment_source": "comment"}
            for index in range(4)
        ],
        "test": [
            {"probe_id": f"test_{index}", "comment_text": f"comment {index}", "comment_source": "comment"}
            for index in range(4)
        ],
    }
    for split_name, payload_rows in rows.items():
        _write_jsonl(input_root / f"{split_name}.jsonl", payload_rows)

    config = CommentRetrievalEvalConfig(
        train_limit=4,
        val_limit=2,
        test_limit=2,
        salt="week10_test_salt",
        require_non_empty_comment=True,
        stratify_by="comment_source",
    )
    first = build_comment_retrieval_eval_regime(
        input_root=input_root,
        output_root=tmp_path / "run1",
        config=config,
    )
    second = build_comment_retrieval_eval_regime(
        input_root=input_root,
        output_root=tmp_path / "run2",
        config=config,
    )
    for split_name in ("train", "val", "test"):
        first_ids = _read_probe_ids(tmp_path / "run1" / "probe_subset" / f"annotated_sidecar_{split_name}.jsonl")
        second_ids = _read_probe_ids(tmp_path / "run2" / "probe_subset" / f"annotated_sidecar_{split_name}.jsonl")
        assert first_ids == second_ids

    manifest = yaml.safe_load((tmp_path / "run1" / "retrieval_eval_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["evaluation_mode"] == "fixed_stratified_subset"
    assert manifest["config"]["salt"] == "week10_test_salt"
    assert manifest["splits"]["train"]["eligible_rows"] == 8
    assert manifest["splits"]["train"]["selected_rows"] == 4
    assert manifest["splits"]["train"]["subset_stratum_counts"]["comment"] >= 1
    assert manifest["splits"]["train"]["subset_stratum_counts"]["comment+nag"] >= 1
    assert first["subset_root"] != second["subset_root"]
