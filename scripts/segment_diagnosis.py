"""진단 ②: move 비언급 세그먼트의 단어 수준 상한/하한 측정.

세그먼트 = 자기 코멘트에 자기 수의 SAN/UCI가 등장하지 않는 pair (~43%).
질문: 이 코멘트들이 전달하는 단어 수준 정보(기물/잡은 기물/전술 플래그/square 언급)로
retrieval이 어디까지 오를 수 있는가 — 상한이 낮으면 세그먼트의 벽은 데이터,
높으면 표현(단어 특징 확장 or text encoder fine-tune)이 레버다.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from modalchess.align.metrics import _ranks_strict  # noqa: E402
from modalchess.align.oracle_ceiling import (  # noqa: E402
    load_pool_rows,
    mention_score_matrix,
    move_attributes,
    symbolic_score_matrix,
)
from modalchess.align.text_embed import normalize_comment  # noqa: E402

KS = (1, 5, 10, 50)

_PIECE_WORDS = {
    "pawn": 1, "knight": 2, "bishop": 3, "rook": 4, "queen": 5, "king": 6,
}
_CAPTURE_RE = re.compile(r"\b(takes?|taking|took|captures?|capturing|captured)\b")
_MOTIF_RE = re.compile(
    r"\b(fork|pin|skewer|sacrifice|sac|discovered|zugzwang|threat|attack|defen[cds]e?\w*|"
    r"develop\w*|castl\w*|promot\w*|blunder|mistake|exchange)\b"
)
_SQUARE_RE = re.compile(r"\b[a-h][1-8]\b")


def _text_word_profile(text: str) -> dict:
    pieces = {code for word, code in _PIECE_WORDS.items() if re.search(rf"\b{word}s?\b", text)}
    return {
        "pieces": pieces,
        "capture": bool(_CAPTURE_RE.search(text)),
        "check": "+" in text or bool(re.search(r"\bcheck\b", text)),
        "mate": "#" in text or bool(re.search(r"\b(checkmate|mate)\b", text)),
        "castle": bool(re.search(r"\b(castl\w*|o-o)\b", text)) or "0-0" in text,
        "promote": bool(re.search(r"\bpromot\w*\b", text)),
        "has_motif": bool(_MOTIF_RE.search(text)),
        "has_square": bool(_SQUARE_RE.search(text)),
    }


def _word_match_scores(profiles: list[dict], attrs: list[dict | None]) -> torch.Tensor:
    """구성적 하한: 코멘트 단어와 후보 board 속성의 일치 개수."""
    n = len(profiles)
    scores = torch.zeros((n, n))
    board_cols = []
    for a in attrs:
        if a is None:
            board_cols.append(None)
            continue
        board_cols.append({
            "pieces": {a["piece_type"]} | ({a["captured_piece_type"]} if a["captured_piece_type"] else set()),
            "capture": a["is_capture"],
            "check": a["gives_check"],
            "mate": a["is_mate"],
            "castle": a["is_castling"],
            "promote": a["is_promotion"],
        })
    for i, prof in enumerate(profiles):
        for j, col in enumerate(board_cols):
            if col is None:
                continue
            s = float(len(prof["pieces"] & col["pieces"]))
            for key in ("capture", "check", "mate", "castle", "promote"):
                if prof[key] and col[key]:
                    s += 1.0
            scores[i, j] = s
    return scores


def _subset_metrics(ranks: torch.Tensor, subset: list[int]) -> dict[str, float]:
    r = ranks[torch.tensor(subset, dtype=torch.long)]
    out = {"mrr": float((1.0 / r).mean())}
    out.update({f"r@{k}": float((r <= k).float().mean()) for k in KS})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="outputs/scale_v1/gate2_comment/corpus/annotated_sidecar_test.jsonl")
    parser.add_argument("--test-board", default="outputs/scale_v1/gate2_comment/embedding_exports/g3/seed11/annotated_sidecar_test_embeddings.pt")
    parser.add_argument("--test-text", default="outputs/connector_v1/text_embeddings/annotated_sidecar_test_text.pt")
    parser.add_argument("--connector", default="outputs/connector_hybrid_v1/rerun_mergedsampler/hybrid_p128/seed11/connector.pt")
    parser.add_argument("--test-features", default="outputs/connector_hybrid_v1/symbolic_features/annotated_sidecar_test_features.pt")
    parser.add_argument("--output", default="outputs/connector_hybrid_v1/segment_diagnosis.json")
    args = parser.parse_args()

    rows = load_pool_rows(args.corpus, args.test_board, args.test_text)
    comments = [normalize_comment(str(r.get("comment_text", ""))) for r in rows]
    attrs = [move_attributes(str(r["fen"]), str(r["target_move_uci"])) for r in rows]
    families = [str(r.get("source_family", "?")) for r in rows]
    fam_group = ["waterhorse" if f.startswith("waterhorse") else f for f in families]

    mention = mention_score_matrix(comments, attrs)
    diag = torch.arange(len(rows))
    segment = [i for i in range(len(rows)) if mention[i, i] == 0.0]
    print(f"pool n={len(rows)}  비언급 세그먼트 n={len(segment)} ({len(segment)/len(rows):.1%})")
    seg_fams = defaultdict(int)
    for i in segment:
        seg_fams[fam_group[i]] += 1
    print("  세그먼트 family:", dict(sorted(seg_fams.items(), key=lambda kv: -kv[1])))

    profiles = [_text_word_profile(comments[i]) for i in range(len(rows))]
    stats = defaultdict(int)
    for i in segment:
        p = profiles[i]
        stats["piece_word"] += bool(p["pieces"])
        stats["capture_verb"] += p["capture"]
        stats["check_or_mate"] += p["check"] or p["mate"]
        stats["motif_word"] += p["has_motif"]
        stats["square_token"] += p["has_square"]
    n_seg = len(segment)
    print("\n[세그먼트 코멘트 단어 함유율]")
    for key, count in stats.items():
        print(f"  {key:<14} {count/n_seg:.3f}")

    report = {"n_pool": len(rows), "n_segment": n_seg, "word_stats": {k: v / n_seg for k, v in stats.items()}}

    print("\n[oracle 상한 — 세그먼트 쿼리, 전체 3000 gallery, t2b]")
    report["oracle_segment"] = {}
    for variant in ("word_level", "word_level_plus_to", "flags_only", "move_plus_flags"):
        scores = symbolic_score_matrix(attrs, variant)
        ranks = _ranks_strict(scores, diag).float()
        m = _subset_metrics(ranks, segment)
        report["oracle_segment"][variant] = m
        print(f"  {variant:<20} mrr={m['mrr']:.4f}  r@10={m['r@10']:.4f}  r@50={m['r@50']:.4f}")

    word_scores = _word_match_scores(profiles, attrs)
    ranks = _ranks_strict(word_scores, diag).float()
    m = _subset_metrics(ranks, segment)
    report["word_match_baseline_segment"] = m
    print(f"\n[구성적 하한 — 단어 일치 매칭, 세그먼트] mrr={m['mrr']:.4f}  r@10={m['r@10']:.4f}  r@50={m['r@50']:.4f}")

    # square-mention 하한: 언급된 bare square ∩ 후보의 {from,to} — square 채널 소진 여부 판별
    import chess as _chess
    mentioned = [
        {_chess.parse_square(s) for s in _SQUARE_RE.findall(comments[i])}
        for i in range(len(rows))
    ]
    move_squares = [
        {a["to_square"], _chess.Move.from_uci(a["uci"]).from_square} if a else set()
        for a in attrs
    ]
    sq_scores = torch.zeros((len(rows), len(rows)))
    for i, ment in enumerate(mentioned):
        if not ment:
            continue
        for j, ms in enumerate(move_squares):
            overlap = len(ment & ms)
            if overlap:
                sq_scores[i, j] = float(overlap)
    ranks = _ranks_strict(sq_scores, diag).float()
    m = _subset_metrics(ranks, segment)
    report["square_match_baseline_segment"] = m
    print(f"[구성적 하한 — square 언급 매칭, 세그먼트] mrr={m['mrr']:.4f}  r@10={m['r@10']:.4f}  r@50={m['r@50']:.4f}")

    # 현 hybrid connector의 세그먼트 실측
    from modalchess.align.dataset import load_aligned_pairs
    from modalchess.align.eval_connector import load_connector
    model, payload = load_connector(args.connector)
    pairs = load_aligned_pairs(
        args.test_board, args.test_text, pool=payload.get("pool", "board_pooled"),
        features_path=args.test_features, feature_mode=payload.get("feature_mode", "hybrid"),
    )
    with torch.no_grad():
        zb = model.encode_board(pairs.board)
        zt = model.encode_text(pairs.text)
    ranks = _ranks_strict(zt @ zb.T, diag).float()
    m = _subset_metrics(ranks, segment)
    report["hybrid_connector_segment"] = m
    print(f"[현 hybrid connector, 세그먼트 실측]      mrr={m['mrr']:.4f}  r@10={m['r@10']:.4f}  r@50={m['r@50']:.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
