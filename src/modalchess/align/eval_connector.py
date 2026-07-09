"""학습된 connector를 test pool에서 채점: strict R@k 양방향 + global/within-family null.

frozen-probe 기준선(new+sentence t2b 0.01084)과 비교해 Gate 4 신호를 낸다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from modalchess.align.connector import AlignmentConnector, ConnectorConfig
from modalchess.align.dataset import load_aligned_pairs
from modalchess.align.metrics import null_control


FROZEN_PROBE_T2B_MRR = 0.01084  # scale_v1 옵션 B, new+sentence, comment regime


def load_connector(connector_path: str | Path) -> tuple[AlignmentConnector, dict[str, Any]]:
    payload = torch.load(connector_path, map_location="cpu", weights_only=False)
    cfg = ConnectorConfig(**payload["model_config"])
    model = AlignmentConnector(cfg)
    model.load_state_dict(payload["connector_state_dict"])
    model.eval()
    return model, payload


def evaluate_connector(config: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_connector(config["connector"])
    model = model.to(device)
    pool = payload.get("pool", config.get("pool", "board_pooled"))
    test_pairs = load_aligned_pairs(config["test_board"], config["test_text"], pool=pool)
    with torch.no_grad():
        zb = model.encode_board(test_pairs.board.to(device)).cpu()
        zt = model.encode_text(test_pairs.text.to(device)).cpu()

    result = null_control(
        zb, zt, test_pairs.source_family,
        repeats=int(config.get("null_repeats", 50)),
        seed=int(config.get("null_seed", 20260710)),
    )
    real = result["real"]
    gnull = result["global_null"]
    wnull = result["within_family_null"]

    t2b = real["text_to_board"]["mrr"]
    b2t = real["board_to_text"]["mrr"]
    verdict = {
        "t2b_mrr": t2b,
        "b2t_mrr": b2t,
        "t2b_over_frozen_probe": t2b / FROZEN_PROBE_T2B_MRR,
        "t2b_above_global_null": t2b > gnull["text_to_board_mrr_max"],
        "t2b_above_within_family_null": t2b > wnull["text_to_board_mrr_max"],
        "b2t_above_within_family_null": b2t > wnull["board_to_text_mrr_max"],
        "beats_frozen_probe_min_bar": t2b >= 1.3 * FROZEN_PROBE_T2B_MRR,
    }
    output = {
        "n_test": int(zb.size(0)),
        "pool": pool,
        "frozen_probe_t2b_mrr": FROZEN_PROBE_T2B_MRR,
        "real": real,
        "global_null": gnull,
        "within_family_null": wnull,
        "verdict": verdict,
    }
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "connector_eval.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output
