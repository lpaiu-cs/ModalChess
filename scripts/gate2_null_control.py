"""Gate 2 permutation null control: real vs shuffled retrieval on the frozen comment regime.

동결된 raw_text_retrieval probe 내부 함수를 재사용해, 각 (backbone, seed, pool, probe_model)에서
test 정렬을 K회 무작위 치환(permutation)했을 때의 strict MRR 분포를 real MRR과 비교한다.
real MRR이 shuffled 분포(null)를 명확히 상회하면 신호가 우연이 아님을 뜻한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from modalchess.eval.raw_text_retrieval import (  # noqa: E402
    _align_rows_by_probe_id,
    _build_vocab,
    _chunked_retrieval_metrics_with_ties,
    _documents_for_family,
    _load_embedding_payload,
    _load_jsonl,
    _normalize_rows,
    _standardize_features,
    _tfidf_matrix,
    _train_text_probe,
)


def _aligned_docs(corpus_root: Path, family: str):
    rows = {sp: _load_jsonl(corpus_root / f"{family}_{sp}.jsonl") for sp in ("train", "val", "test")}
    targets = {
        sp: (_load_jsonl(corpus_root / f"{family}_targets_{sp}.jsonl")
             if (corpus_root / f"{family}_targets_{sp}.jsonl").exists() else None)
        for sp in ("train", "val", "test")
    }
    probe_ids, docs = {}, {}
    for sp in ("train", "val", "test"):
        ar, at = _align_rows_by_probe_id(rows[sp], targets[sp])
        probe_ids[sp] = [str(r["probe_id"]) for r in ar]
        docs[sp] = _documents_for_family(family, ar, at)
    return probe_ids, docs


def _tfidf_targets(docs, min_df, max_vocab):
    vocab, tok2idx, idf = _build_vocab(docs["train"], min_df=min_df, max_vocab_size=max_vocab)
    return {sp: _normalize_rows(_tfidf_matrix(docs[sp], tok2idx, idf)) for sp in ("train", "val", "test")}


def _sentence_targets(docs, model_name: str, batch_size: int):
    """문장 인코더(mean-pooled, normalized)로 텍스트 target 행렬을 만든다."""
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).eval().to(device)

    def encode_all(texts: list[str]) -> torch.Tensor:
        out = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                hidden = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            out.append(torch.nn.functional.normalize(emb, dim=1).cpu())
        return torch.cat(out, dim=0)

    return {sp: encode_all(docs[sp]) for sp in ("train", "val", "test")}


def _prepare(corpus_root: Path, family: str, args):
    probe_ids, docs = _aligned_docs(corpus_root, family)
    if args.text_side == "sentence":
        targets = _sentence_targets(docs, args.sentence_model, args.sentence_batch)
    else:
        targets = _tfidf_targets(docs, args.min_df, args.max_vocab)
    return probe_ids, targets


def _features(embedding_dir: Path, family: str, pool: str, probe_ids: dict[str, list[str]]):
    feats = {}
    for sp in ("train", "val", "test"):
        payload = _load_embedding_payload(embedding_dir / f"{family}_{sp}_embeddings.pt")
        idx = {str(p): i for i, p in enumerate(payload["probe_id"])}
        order = torch.tensor([idx[p] for p in probe_ids[sp]], dtype=torch.long)
        feats[sp] = payload[pool].index_select(0, order).float()
    return _standardize_features(feats["train"], feats["val"], feats["test"])


def _mrr_pair(predicted_test, tfidf_test):
    b2t = _chunked_retrieval_metrics_with_ties(predicted_test, tfidf_test, tie_mode="strict")
    t2b = _chunked_retrieval_metrics_with_ties(tfidf_test, predicted_test, tie_mode="strict")
    return b2t[2], t2b[2], b2t[0], t2b[0]  # b2t_mrr, t2b_mrr, b2t_r1, t2b_r1


def run(args) -> None:
    corpus_root = Path(args.corpus_root)
    probe_ids, tfidf = _prepare(corpus_root, args.family, args)
    n_test = tfidf["test"].size(0)
    records = []
    for label, emb_root in (("old", Path(args.old_embedding_root)), ("new", Path(args.new_embedding_root))):
        for bb in args.backbones:
            for seed in args.seeds:
                edir = emb_root / bb / f"seed{seed}"
                if not (edir / f"{args.family}_test_embeddings.pt").exists():
                    continue
                tr, va, te = _features(edir, args.family, args.pool, probe_ids)
                model, _ = _train_text_probe(
                    model_kind=args.probe_model,
                    train_features=tr, train_targets=tfidf["train"],
                    val_features=va, val_targets=tfidf["val"],
                    seed=seed, max_train_rows=(None if args.probe_model == "linear" else 50000),
                )
                model.eval()
                with torch.no_grad():
                    predicted = _normalize_rows(model(te))
                real_b2t, real_t2b, real_b2t_r1, real_t2b_r1 = _mrr_pair(predicted, tfidf["test"])
                # permutation null: shuffle predicted rows, breaking true pairing
                nb2t, nt2b = [], []
                g = torch.Generator().manual_seed(args.null_seed + seed)
                for _ in range(args.null_repeats):
                    perm = torch.randperm(predicted.size(0), generator=g)
                    sb2t, st2b, _, _ = _mrr_pair(predicted.index_select(0, perm), tfidf["test"])
                    nb2t.append(sb2t); nt2b.append(st2b)
                rec = {
                    "backbone_group": label, "backbone": bb, "seed": seed,
                    "text_side": args.text_side,
                    "pool": args.pool, "probe_model": args.probe_model, "n_test": n_test,
                    "real_b2t_mrr": real_b2t, "real_t2b_mrr": real_t2b,
                    "real_b2t_r1": real_b2t_r1, "real_t2b_r1": real_t2b_r1,
                    "null_b2t_mrr_mean": st.mean(nb2t), "null_b2t_mrr_max": max(nb2t),
                    "null_t2b_mrr_mean": st.mean(nt2b), "null_t2b_mrr_max": max(nt2b),
                }
                records.append(rec)
                print(f"{label:3} {bb} s{seed} {args.pool[:4]}/{args.probe_model[:3]} | "
                      f"real b2t {real_b2t:.5f} t2b {real_t2b:.5f} | "
                      f"null(mean/max) b2t {st.mean(nb2t):.5f}/{max(nb2t):.5f} t2b {st.mean(nt2b):.5f}/{max(nt2b):.5f}")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "gate2_null_control.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'gate2_null_control.json'}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-root", default="outputs/scale_v1/gate2_comment/corpus")
    p.add_argument("--old-embedding-root", default="outputs/week17/embedding_exports/current_mixed_baseline")
    p.add_argument("--new-embedding-root", default="outputs/scale_v1/gate2_comment/embedding_exports")
    p.add_argument("--output-dir", default="outputs/scale_v1/gate2_comment/null_control")
    p.add_argument("--family", default="annotated_sidecar")
    p.add_argument("--pool", default="board_pooled")
    p.add_argument("--probe-model", default="mlp")
    p.add_argument("--backbone", dest="backbones", action="append", default=[])
    p.add_argument("--seed", dest="seeds", type=int, action="append", default=[])
    p.add_argument("--text-side", choices=("tfidf", "sentence"), default="tfidf")
    p.add_argument("--sentence-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--sentence-batch", type=int, default=256)
    p.add_argument("--min-df", type=int, default=25)
    p.add_argument("--max-vocab", type=int, default=512)
    p.add_argument("--null-repeats", type=int, default=50)
    p.add_argument("--null-seed", type=int, default=20260710)
    a = p.parse_args()
    if not a.backbones:
        a.backbones = ["g1", "g3"]
    if not a.seeds:
        a.seeds = [11, 17, 23]
    return a


if __name__ == "__main__":
    run(parse_args())
