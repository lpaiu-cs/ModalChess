"""Oracle ceiling: 주어진 pair 데이터가 허용하는 retrieval 상한을 심볼릭하게 잰다.

Phase 1 진단 1순위 — connector의 낮은 절대 retrieval(R@50 ~8%)이
(A) 인코더 병목인지 (B) 데이터 모호성(generic comment) 병목인지 판별한다.

세 종류를 계산한다 (모두 gate2/connector와 동일한 strict tie rank):
1. duplicate ceiling — 어떤 함수도 못 넘는 상한.
   - assignment: 동일 쿼리(같은 normalized_text 등) k개 행은 단일 랭킹을 공유
     → 최적이어도 rank {1..k} 분배 → MRR 상한 = mean H(k)/k.
   - tie: 정답과 동일한 gallery 항목(같은 fen 등) j개는 어떤 함수든 동점
     → strict rank >= j.
2. symbolic attribute oracle — 코멘트가 정답 board의 (수, 전술 플래그)를 완벽히
   전달한다고 가정했을 때의 상한. 이게 낮으면 "완벽한 move-conditioned 코멘트"라도
   pool 모호성 때문에 retrieval이 안 되는 것 → 데이터가 벽.
3. mention baseline — 후보 board의 SAN/UCI 문자열이 쿼리 코멘트에 실제로 등장하는지로
   채점하는 구성적 하한. 학습 없이 심볼릭 매칭만으로 도달 가능한 성능.

주의: tie ceiling의 board dup 키는 fen이다. board encoder 입력이 fen 단독이라는
가정 하의 상한이며, history 조건부 인코더에는 보수적이지 않을 수 있다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

import chess
import torch

from modalchess.align.dataset import load_aligned_pairs
from modalchess.align.metrics import _ranks_strict
from modalchess.align.text_embed import normalize_comment

KS = (1, 5, 10, 50)

_CASTLE_TOKENS = ("o-o-o", "0-0-0", "o-o", "0-0")


def move_attributes(fen: str, uci: str) -> dict[str, Any] | None:
    """fen+uci에서 심볼릭 속성(SAN, 기물, 전술 플래그)을 유도한다. 비합법 수면 None."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return None
        san = board.san(move)
        attrs = {
            "uci": uci,
            "san": san,
            "piece_type": int(board.piece_type_at(move.from_square) or 0),
            "is_capture": bool(board.is_capture(move)),
            "is_castling": bool(board.is_castling(move)),
            "is_promotion": move.promotion is not None,
            "side_to_move": board.turn,
        }
        board.push(move)
        attrs["gives_check"] = bool(board.is_check())
        attrs["is_mate"] = bool(board.is_checkmate())
        return attrs
    except (ValueError, AssertionError):
        return None


# ---------------------------------------------------------------------------
# 1. duplicate ceiling
# ---------------------------------------------------------------------------

def assignment_ceiling(query_keys: list[str], ks: tuple[int, ...] = KS) -> dict[str, float]:
    """동일 쿼리 키 k개 행은 랭킹 하나를 공유 → 최적 rank {1..k}일 때의 상한."""
    counts = Counter(query_keys)
    n = len(query_keys)
    mrr_sum = 0.0
    r_at = {k: 0.0 for k in ks}
    for size in counts.values():
        mrr_sum += sum(1.0 / r for r in range(1, size + 1))  # H(size)
        for k in ks:
            r_at[k] += min(size, k)
    out = {"mrr": mrr_sum / n}
    out.update({f"r@{k}": r_at[k] / n for k in ks})
    return out


def tie_ceiling(target_dup_counts: list[int], ks: tuple[int, ...] = KS) -> dict[str, float]:
    """정답과 동일한 gallery 항목 j개(자기 포함) → strict rank >= j일 때의 상한."""
    ranks = torch.tensor(target_dup_counts, dtype=torch.float)
    out = {"mrr": float((1.0 / ranks).mean())}
    out.update({f"r@{k}": float((ranks <= k).float().mean()) for k in ks})
    return out


def duplicate_ceiling(
    query_keys: list[str],
    gallery_keys: list[str],
    ks: tuple[int, ...] = KS,
) -> dict[str, Any]:
    """assignment(쿼리 dup) + tie(gallery dup) 상한과 그 min(둘 다 유효한 상한)."""
    gallery_counts = Counter(gallery_keys)
    dup_of_target = [gallery_counts[key] for key in gallery_keys]
    assign = assignment_ceiling(query_keys, ks)
    tie = tie_ceiling(dup_of_target, ks)
    combined = {name: min(assign[name], tie[name]) for name in assign}
    return {"assignment": assign, "tie": tie, "combined_min": combined}


# ---------------------------------------------------------------------------
# 2. symbolic attribute oracle
# ---------------------------------------------------------------------------

_FLAG_FIELDS = (
    "piece_type", "is_capture", "is_castling", "is_promotion",
    "side_to_move", "gives_check", "is_mate",
)
_MOVE_WEIGHT = 10.0  # flag 합(최대 7)보다 커야 move 일치가 사전식으로 우선한다


def _encode_column(values: list[Any]) -> torch.Tensor:
    codebook: dict[Any, int] = {}
    codes = []
    for value in values:
        codes.append(codebook.setdefault(value, len(codebook)))
    return torch.tensor(codes, dtype=torch.long)


def symbolic_score_matrix(attrs: list[dict[str, Any] | None], variant: str) -> torch.Tensor:
    """행 i의 정답 속성과 후보 j의 속성 일치도 행렬. None 속성은 매칭 불가 sentinel."""
    def column(field: str) -> list[Any]:
        return [a[field] if a is not None else f"__none_{i}" for i, a in enumerate(attrs)]

    n = len(attrs)
    scores = torch.zeros((n, n), dtype=torch.float)
    if variant in {"uci_exact", "san_exact"}:
        codes = _encode_column(column("uci" if variant == "uci_exact" else "san"))
        return (codes.unsqueeze(1) == codes.unsqueeze(0)).float()
    if variant in {"move_plus_flags", "flags_only"}:
        for field in _FLAG_FIELDS:
            codes = _encode_column(column(field))
            scores += (codes.unsqueeze(1) == codes.unsqueeze(0)).float()
        if variant == "move_plus_flags":
            codes = _encode_column(column("uci"))
            scores += _MOVE_WEIGHT * (codes.unsqueeze(1) == codes.unsqueeze(0)).float()
        return scores
    raise ValueError(f"unknown variant: {variant}")


def metrics_from_scores(scores: torch.Tensor, ks: tuple[int, ...] = KS) -> dict[str, Any]:
    """score 행렬에서 양방향 strict 지표 + per-row rank."""
    n = scores.size(0)
    diag = torch.arange(n)
    out = {}
    ranks_by_direction = {}
    for name, matrix in (("text_to_board", scores), ("board_to_text", scores.transpose(0, 1))):
        ranks = _ranks_strict(matrix, diag).float()
        metrics = {"mrr": float((1.0 / ranks).mean())}
        metrics.update({f"r@{k}": float((ranks <= k).float().mean()) for k in ks})
        out[name] = metrics
        ranks_by_direction[name] = ranks
    return {"metrics": out, "ranks": ranks_by_direction}


# ---------------------------------------------------------------------------
# 3. mention baseline
# ---------------------------------------------------------------------------

def _mention_keys(attrs: dict[str, Any] | None) -> tuple[set[str], bool]:
    """후보 move를 코멘트에서 찾을 때 쓸 토큰 키들 + 캐슬링 여부."""
    if attrs is None:
        return set(), False
    san_core = re.sub(r"[+#]+$", "", attrs["san"].lower())
    keys = {attrs["uci"].lower()}
    if attrs["is_castling"]:
        return keys, True
    keys.add(re.sub(r"[^a-z0-9]", "", san_core))
    return keys, False


def mention_score_matrix(comments_normalized: list[str], attrs: list[dict[str, Any] | None]) -> torch.Tensor:
    """S[i,j] = 후보 j의 수(SAN/UCI 토큰)가 코멘트 i에 등장하면 1."""
    token_sets = [set(re.findall(r"[a-z0-9]+", text)) for text in comments_normalized]
    has_castle = [any(tok in text for tok in _CASTLE_TOKENS) for text in comments_normalized]
    keys_and_castle = [_mention_keys(a) for a in attrs]

    # 후보를 키별로 그룹화해 (unique key) x (row) 만 검사한다.
    rows_by_key: dict[str, list[int]] = defaultdict(list)
    castle_cols: list[int] = []
    for j, (keys, is_castle) in enumerate(keys_and_castle):
        if is_castle:
            castle_cols.append(j)
        for key in keys:
            rows_by_key[key].append(j)

    n = len(comments_normalized)
    scores = torch.zeros((n, n), dtype=torch.float)
    for i, tokens in enumerate(token_sets):
        for token in tokens:
            for j in rows_by_key.get(token, ()):
                scores[i, j] = 1.0
        if has_castle[i]:
            for j in castle_cols:
                scores[i, j] = 1.0
    return scores


# ---------------------------------------------------------------------------
# 실행: gate4와 동일한 test pool 위에서 전체 리포트
# ---------------------------------------------------------------------------

def _aggregate_by_group(ranks: torch.Tensor, groups: list[str], ks: tuple[int, ...] = KS) -> dict[str, dict[str, float]]:
    by_group: dict[str, list[float]] = defaultdict(list)
    for rank, group in zip(ranks.tolist(), groups):
        by_group[group].append(rank)
    out = {}
    for group, values in sorted(by_group.items(), key=lambda kv: -len(kv[1])):
        t = torch.tensor(values)
        entry = {"n": len(values), "mrr": float((1.0 / t).mean())}
        entry.update({f"r@{k}": float((t <= k).float().mean()) for k in ks})
        out[group] = entry
    return out


def load_pool_rows(
    corpus_path: str | Path,
    board_embedding_path: str | Path,
    text_embedding_path: str | Path,
    pool: str = "board_pooled",
) -> list[dict[str, Any]]:
    """gate4 eval과 동일한 pool(probe_id 교집합·순서)로 corpus 행을 정렬한다."""
    pairs = load_aligned_pairs(board_embedding_path, text_embedding_path, pool=pool)
    rows_by_probe: dict[str, dict[str, Any]] = {}
    with open(corpus_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows_by_probe[str(row["probe_id"])] = row
    missing = [pid for pid in pairs.probe_id if pid not in rows_by_probe]
    if missing:
        raise ValueError(f"corpus에 없는 pool probe_id {len(missing)}개 (예: {missing[:3]})")
    return [rows_by_probe[pid] for pid in pairs.probe_id]


def run_oracle_ceiling(config: dict[str, Any]) -> dict[str, Any]:
    rows = load_pool_rows(
        config["corpus"], config["test_board"], config["test_text"],
        pool=config.get("pool", "board_pooled"),
    )
    comments = [normalize_comment(str(row.get("comment_text", ""))) for row in rows]
    fens = [str(row["fen"]) for row in rows]
    families = [str(row.get("source_family", "unknown")) for row in rows]
    buckets = [str(row.get("informativeness_bucket", "unknown")) for row in rows]
    attrs = [move_attributes(str(row["fen"]), str(row["target_move_uci"])) for row in rows]
    n_invalid = sum(1 for a in attrs if a is None)

    report: dict[str, Any] = {
        "n_pool": len(rows),
        "n_invalid_moves": n_invalid,
        "duplicate_ceiling": {
            "text_to_board": duplicate_ceiling(query_keys=comments, gallery_keys=fens),
            "board_to_text": duplicate_ceiling(query_keys=fens, gallery_keys=comments),
        },
        "symbolic_oracle": {},
        "mention_baseline": {},
    }

    breakdown_ranks: dict[str, torch.Tensor] = {}
    for variant in ("uci_exact", "san_exact", "move_plus_flags", "flags_only"):
        result = metrics_from_scores(symbolic_score_matrix(attrs, variant))
        report["symbolic_oracle"][variant] = result["metrics"]
        if variant == "move_plus_flags":
            breakdown_ranks["oracle_move_plus_flags_t2b"] = result["ranks"]["text_to_board"]

    mention_scores = mention_score_matrix(comments, attrs)
    mention_result = metrics_from_scores(mention_scores)
    report["mention_baseline"] = mention_result["metrics"]
    breakdown_ranks["mention_t2b"] = mention_result["ranks"]["text_to_board"]

    # 자기 pair의 수가 자기 코멘트에 등장하는 비율 = move-conditioned fraction
    diag = torch.arange(len(rows))
    self_mention = mention_scores[diag, diag]
    report["move_conditioned_fraction"] = float(self_mention.mean())
    report["move_conditioned_by_family"] = {
        family: {"n": len(vals), "fraction": float(torch.tensor(vals).mean())}
        for family, vals in sorted(
            _group_values(self_mention.tolist(), families).items(),
            key=lambda kv: -len(kv[1]),
        )
    }

    report["breakdown"] = {
        name: {
            "by_source_family": _aggregate_by_group(ranks, families),
            "by_informativeness_bucket": _aggregate_by_group(ranks, buckets),
        }
        for name, ranks in breakdown_ranks.items()
    }

    output_dir = Path(config.get("output_dir", "outputs/connector_v1/oracle_ceiling"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "oracle_ceiling.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _group_values(values: list[float], groups: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        out[group].append(value)
    return out
